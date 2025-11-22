import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageDraw, ImageFont
import sys
import gc
import os
import json
import pygame

# ========== 配置存储路径 ==========
CONFIG_PATH = "snake_game_config.json"

# ========== 默认音频文件名 ==========
BGM_FILE = "bgm.mp3"
EAT_LEFT_FILE = "eat_left.wav"
EAT_RIGHT_FILE = "eat_right.wav"
DEAD_FILE = "dead.wav"

# ========== 摄像头初始化 ==========
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头无法打开，请检查连接/权限/是否被其他程序占用。")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
ret, img = cap.read()
if not ret or img is None:
    print("⚠️ 无法捕获摄像头画面。")
    cap.release()
    sys.exit(1)
else:
    print("摄像头初始化成功，准备开始游戏!")

# ========== 配置文件自动保存/读取 ==========
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

game_cfg = load_config()
default_food_img = game_cfg.get("food_img_path", "xin.png")
default_left_head = game_cfg.get("left_head_path", "")
default_right_head = game_cfg.get("right_head_path", "")
BGM_FILE = game_cfg.get("bgm_file", BGM_FILE)
EAT_LEFT_FILE = game_cfg.get("eat_left_file", EAT_LEFT_FILE)
EAT_RIGHT_FILE = game_cfg.get("eat_right_file", EAT_RIGHT_FILE)
DEAD_FILE = game_cfg.get("dead_file", DEAD_FILE)

detector = HandDetector(detectionCon=0.7, maxHands=2)

# ========== 音乐与音效切换/加载封装 ==========
pygame.mixer.init()

def load_bgm(bgm_file):
    if not os.path.isfile(bgm_file):
        print(f"❌ 未找到背景音乐文件: {bgm_file}")
        return False
    try:
        pygame.mixer.music.load(bgm_file)
        pygame.mixer.music.set_volume(0.4)
        print(f"🎵 已设置背景音乐：{bgm_file}")
        return True
    except Exception as e:
        print(f"❌ 背景音乐加载失败: {e}")
        return False

def load_eat_sound(eat_file):
    if not os.path.isfile(eat_file):
        print(f"❌ 未找到音效文件: {eat_file}")
        return None
    try:
        eat_sound = pygame.mixer.Sound(eat_file)
        eat_sound.set_volume(0.8)
        print(f"🍎 已设置音效：{eat_file}")
        return eat_sound
    except Exception as e:
        print(f"❌ 音效加载失败: {e}")
        return None

def load_dead_sound(dead_file):
    if not os.path.isfile(dead_file):
        print(f"❌ 未找到死亡音效文件: {dead_file}")
        return None
    try:
        dead_sound = pygame.mixer.Sound(dead_file)
        dead_sound.set_volume(1.0)
        print(f"💀 已设置死亡音效：{dead_file}")
        return dead_sound
    except Exception as e:
        print(f"❌ 死亡音效加载失败: {e}")
        return None

def switch_bgm(new_file):
    pygame.mixer.music.stop()
    if load_bgm(new_file):
        pygame.mixer.music.play(-1)

def switch_left_eat_sound(new_file):
    global eat_sound_left
    eat_sound_left = load_eat_sound(new_file)
    if hasattr(game, "eat_sound_left"):
        game.eat_sound_left = eat_sound_left

def switch_right_eat_sound(new_file):
    global eat_sound_right
    eat_sound_right = load_eat_sound(new_file)
    if hasattr(game, "eat_sound_right"):
        game.eat_sound_right = eat_sound_right

def switch_dead_sound(new_file):
    global dead_sound
    dead_sound = load_dead_sound(new_file)
    if hasattr(game, "dead_sound"):
        game.dead_sound = dead_sound

# 加载初始音乐及音效
load_bgm(BGM_FILE)
eat_sound_left = load_eat_sound(EAT_LEFT_FILE)
eat_sound_right = load_eat_sound(EAT_RIGHT_FILE)
dead_sound = load_dead_sound(DEAD_FILE)

def circle_crop_resize_colormap(img_pil, target_size, base_color_rgb=None):
    img = img_pil.convert("RGBA")
    tw, th = target_size
    img.thumbnail((tw, th), Image.LANCZOS)
    blank = Image.new("RGBA", target_size, (0, 0, 0, 0))
    x = (tw - img.width) // 2
    y = (th - img.height) // 2
    blank.paste(img, (x, y), img)
    mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(mask)
    radius = min(tw, th) // 2 - 2
    draw.ellipse((tw // 2 - radius, th // 2 - radius, tw // 2 + radius, th // 2 + radius), fill=255)
    blank_np = np.array(blank)
    alpha = np.array(mask)
    blank_np[..., 3] = alpha
    if base_color_rgb:
        mask_float = alpha / 255.0
        base_rgb = np.array(base_color_rgb, dtype=np.uint8)
        for c in range(3):
            blank_np[..., c] = (blank_np[..., c].astype(np.float32) * mask_float +
                                base_rgb[c] * (1 - mask_float)).astype(np.uint8)
    return blank_np

class DualSnakeGameClass:
    """
    双人对战贪吃蛇：支持动态食物/蛇头图片、圆形、历史保存、BGM、吃食物音效和死亡音效热替换
    """
    def __init__(self, pathFood, left_head_img, right_head_img, eat_sound_left=None, eat_sound_right=None, dead_sound=None):
        self.food_img_size = (50, 50)
        self.head_img_size = (40, 40)
        self.snakes = {
            "left": {
                "points": [],
                "lengths": [],
                "currentLength": 0,
                "allowedLength": 150,
                "previousHead": (0, 0),
                "score": 0,
                "color": (0, 0, 255),
                "head_color": (200, 0, 200),
                "head_img": None,
                "head_img_path": left_head_img or ""
            },
            "right": {
                "points": [],
                "lengths": [],
                "currentLength": 0,
                "allowedLength": 150,
                "previousHead": (0, 0),
                "score": 0,
                "color": (255, 0, 0),
                "head_color": (0, 200, 200),
                "head_img": None,
                "head_img_path": right_head_img or ""
            }
        }
        self.imgFood = None
        self.wFood = 0
        self.hFood = 0
        self.food_img_path = pathFood or ""
        self.set_food_image(pathFood)
        self.set_head_image("left", left_head_img)
        self.set_head_image("right", right_head_img)
        self.foodPoint = 0, 0
        self.randomFoodLocation()
        self.gameStarted = False
        self.gameOver = False
        self.collisionCooldown = 0
        self.min_collision_distance = 45

        self.eat_sound_left = eat_sound_left
        self.eat_sound_right = eat_sound_right
        self.dead_sound = dead_sound

    def set_food_image(self, pathFood):
        target_w, target_h = self.food_img_size
        self.food_img_path = pathFood or ""
        if pathFood and os.path.isfile(pathFood):
            try:
                img_pil = Image.open(pathFood).convert("RGBA")
                food_np = circle_crop_resize_colormap(img_pil, (target_w, target_h))
                self.imgFood = food_np
                self.hFood, self.wFood, _ = food_np.shape
                print(f"✅ 已成功加载并圆形缩放食物图片：{pathFood}")
                return True
            except Exception as e:
                print(f"⚠️ 食物图片处理失败({e})，使用默认食物")
        blank = np.ones((target_h, target_w, 4), dtype=np.uint8) * 255
        cv2.circle(blank, (target_w // 2, target_h // 2), min(target_w, target_h) // 2 - 5, (0, 255, 0, 255), -1)
        self.imgFood = blank
        self.hFood, self.wFood, _ = self.imgFood.shape
        return False

    def set_head_image(self, which, pathImg):
        target_w, target_h = self.head_img_size
        self.snakes[which]["head_img_path"] = pathImg or ""
        base_color = self.snakes[which]["head_color"]
        if pathImg and os.path.isfile(pathImg):
            try:
                img_pil = Image.open(pathImg).convert("RGBA")
                img_np = circle_crop_resize_colormap(img_pil, (target_w, target_h), base_color)
                self.snakes[which]["head_img"] = img_np
                print(f"✅ {which}蛇头图片已裁剪圆形缩放并融合主色: {pathImg}")
                return True
            except Exception as e:
                print(f"⚠️ 蛇头图片处理失败({e})，使用默认圆形色头")
        self.snakes[which]["head_img"] = None
        return False

    def set_dead_sound(self, sound):
        self.dead_sound = sound

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def put_chinese_text(self, img, text, position, font_size=30, color=(0, 255, 0)):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", font_size)
            except:
                font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def play_eat_sound(self, snake_data):
        if snake_data["color"] == (0, 0, 255) and self.eat_sound_left:
            self.eat_sound_left.play()
        elif snake_data["color"] == (255, 0, 0) and self.eat_sound_right:
            self.eat_sound_right.play()

    def play_dead_sound(self):
        if self.dead_sound:
            self.dead_sound.play()

    def update_snake(self, snake_data, currentHead):
        px, py = snake_data["previousHead"]
        cx, cy = currentHead
        snake_data["points"].append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        snake_data["lengths"].append(distance)
        snake_data["currentLength"] += distance
        snake_data["previousHead"] = cx, cy
        if snake_data["currentLength"] > snake_data["allowedLength"]:
            while snake_data["currentLength"] > snake_data["allowedLength"] and snake_data["lengths"]:
                snake_data["currentLength"] -= snake_data["lengths"].pop(0)
                snake_data["points"].pop(0)
        rx, ry = self.foodPoint
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.randomFoodLocation()
            snake_data["allowedLength"] += 50
            snake_data["score"] += 1
            self.play_eat_sound(snake_data)
            print(f"{'左手' if snake_data['color'] == (0, 0, 255) else '右手'}得分! 当前分数: {snake_data['score']}")

    def draw_snake(self, img, snake_data):
        if len(snake_data["points"]) > 1:
            points_array = np.array(snake_data["points"], dtype=np.int32)
            for i in range(1, len(points_array)):
                cv2.line(img, tuple(points_array[i - 1]), tuple(points_array[i]), snake_data["color"], 20)
            cx, cy = points_array[-1]
            head_img = snake_data["head_img"]
            if head_img is not None:
                h_img, w_img = head_img.shape[0], head_img.shape[1]
                x1 = int(cx - w_img // 2)
                y1 = int(cy - h_img // 2)
                x2 = x1 + w_img
                y2 = y1 + h_img
                if (0 <= x1 < img.shape[1]) and (0 <= y1 < img.shape[0]) and (x2 <= img.shape[1]) and (y2 <= img.shape[0]):
                    roi = img[y1:y2, x1:x2]
                    mask = head_img[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)
                    img_bgr = head_img[:, :, :3]
                    for c in range(3):
                        roi[:, :, c] = (roi[:, :, c] * (mask_inv // 255) + img_bgr[:, :, c] * (mask // 255)).astype(np.uint8)
                    img[y1:y2, x1:x2] = roi
                else:
                    cv2.circle(img, (cx, cy), 20, tuple(snake_data["head_color"]), cv2.FILLED)
            else:
                cv2.circle(img, (cx, cy), 20, tuple(snake_data["head_color"]), cv2.FILLED)

    def check_collision(self, snake_data):
        if len(snake_data["points"]) > 30 and self.collisionCooldown == 0:
            check_points = snake_data["points"][:-20]
            if len(check_points) > 10:
                cx, cy = snake_data["points"][-1]
                pts = np.array(check_points, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                head_point = np.array([[cx, cy]], dtype=np.float32)
                body_points = np.array(check_points, dtype=np.float32)
                distances = np.sqrt(np.sum((body_points - head_point) ** 2, axis=1))
                min_distance = np.min(distances)
                if min_distance < self.min_collision_distance:
                    return True
        return False

    def update(self, imgMain, left_hand_head, right_hand_head):
        imgMain = self.put_chinese_text(
            imgMain,
            f"F:食物 H:左蛇头 J:右蛇头  B:背景音乐  C:左蛇音效  V:右蛇音效  D:死亡音效  Q:退出 空格:开始/重开 +/-:灵敏度",
            (10, 25), font_size=22, color=(255, 220, 20))
        imgMain = self.put_chinese_text(
            imgMain,
            f"食物:{os.path.basename(self.food_img_path) or '默认'}  左头:{os.path.basename(self.snakes['left']['head_img_path']) or '默认'}  右头:{os.path.basename(self.snakes['right']['head_img_path']) or '默认'}",
            (10, 55), font_size=20, color=(0, 180, 255))
        if not self.gameStarted:
            imgMain = self.put_chinese_text(imgMain, "左右手贪吃蛇", (470, 80), font_size=56, color=(0, 255, 0))
            if left_hand_head != (0, 0):
                cv2.circle(imgMain, left_hand_head, 20, (0, 0, 255), cv2.FILLED)
                imgMain = self.put_chinese_text(imgMain, "左手已就绪", (left_hand_head[0] + 30, left_hand_head[1]),
                                                font_size=20, color=(160, 220, 255))
            if right_hand_head != (0, 0):
                cv2.circle(imgMain, right_hand_head, 20, (255, 0, 0), cv2.FILLED)
                imgMain = self.put_chinese_text(imgMain, "右手已就绪", (right_hand_head[0] + 30, right_hand_head[1]),
                                                font_size=20, color=(160, 220, 255))
        elif self.gameOver:
            left_score = self.snakes["left"]["score"]
            right_score = self.snakes["right"]["score"]
            winner = "左手" if left_score > right_score else "右手" if right_score > left_score else "平手"
            imgMain = self.put_chinese_text(imgMain, "GameOver", (480, 300), font_size=60, color=(0, 255, 0))
            imgMain = self.put_chinese_text(imgMain, f'左手得分: {left_score}', (20, 550), font_size=40, color=(0, 255, 0))
            imgMain = self.put_chinese_text(imgMain, f'右手得分: {right_score}', (1000, 550), font_size=40, color=(0, 255, 0))
            imgMain = self.put_chinese_text(imgMain, f'获胜手: {winner}', (470, 520), font_size=40, color=(0, 255, 0))
            imgMain = self.put_chinese_text(imgMain, "按空格键重新开始", (470, 600), font_size=30, color=(0, 255, 0))
        else:
            if left_hand_head != (0, 0):
                self.update_snake(self.snakes["left"], left_hand_head)
            if right_hand_head != (0, 0):
                self.update_snake(self.snakes["right"], right_hand_head)
            self.draw_snake(imgMain, self.snakes["left"])
            self.draw_snake(imgMain, self.snakes["right"])
            rx, ry = self.foodPoint
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))
            left_score = self.snakes["left"]["score"]
            right_score = self.snakes["right"]["score"]
            imgMain = self.put_chinese_text(imgMain, f'左手: {left_score}分', (50, 120), font_size=28, color=(0, 180, 220))
            imgMain = self.put_chinese_text(imgMain, f'右手: {right_score}分', (1000, 120), font_size=28, color=(0, 180, 220))
            if self.collisionCooldown > 0:
                self.collisionCooldown -= 1
            left_collision = len(self.snakes["left"]["points"]) > 30 and self.check_collision(self.snakes["left"])
            right_collision = len(self.snakes["right"]["points"]) > 30 and self.check_collision(self.snakes["right"])
            if left_collision or right_collision:
                # 死亡播放音效
                self.play_dead_sound()
                self.gameOver = True
                self.collisionCooldown = 30
        return imgMain

    def reset(self):
        for snake in self.snakes.values():
            snake["points"] = []
            snake["lengths"] = []
            snake["currentLength"] = 0
            snake["allowedLength"] = 150
            snake["previousHead"] = (0, 0)
            snake["score"] = 0
        self.randomFoodLocation()
        self.gameStarted = False
        self.gameOver = False
        self.collisionCooldown = 0
        print("游戏已重置")

    def start_game(self):
        self.gameStarted = True
        self.gameOver = False
        print("游戏开始!")

def config_save():
    save_config({
        "food_img_path": game.food_img_path,
        "left_head_path": game.snakes["left"]["head_img_path"],
        "right_head_path": game.snakes["right"]["head_img_path"],
        "bgm_file": BGM_FILE,
        "eat_left_file": EAT_LEFT_FILE,
        "eat_right_file": EAT_RIGHT_FILE,
        "dead_file": DEAD_FILE
    })

# ========== 游戏主进程 ==========
game = DualSnakeGameClass(default_food_img, default_left_head, default_right_head, eat_sound_left, eat_sound_right, dead_sound)

try:
    max_fail = 30
    fail_count = 0
    bgm_played = False
    while True:
        success, img = cap.read()
        if not success or img is None:
            fail_count += 1
            print(f"无法从摄像头读取({fail_count}) ...")
            if fail_count > max_fail:
                print(f"摄像头多次失败，程序退出。")
                break
            continue
        fail_count = 0
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False, draw=True)
        left_hand_head = (0, 0)
        right_hand_head = (0, 0)
        if hands:
            for hand in hands:
                lmList = hand['lmList']
                pointIndex = tuple(map(int, lmList[8][0:2]))
                if hand['type'] == 'Left':
                    left_hand_head = pointIndex
                    cv2.circle(img, pointIndex, 15, (0, 0, 255), cv2.FILLED)
                else:
                    right_hand_head = pointIndex
                    cv2.circle(img, pointIndex, 15, (255, 0, 0), cv2.FILLED)
        img = game.update(img, left_hand_head, right_hand_head)
        cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img)
        key = cv2.waitKey(1)
        if key == ord(' '):
            if not game.gameStarted:
                if not bgm_played:
                    try:
                        pygame.mixer.music.play(-1)
                        bgm_played = True
                    except Exception as e:
                        print(f"背景音乐播放失败: {e}")
                game.start_game()
            else:
                pygame.mixer.music.stop()
                bgm_played = False
                game.reset()
        elif key == ord('q') or key == ord('Q'):
            pygame.mixer.music.stop()
            print("按下Q，安全退出。")
            break
        elif key == ord('+') or key == ord('='):
            game.min_collision_distance += 5
            print(f"碰撞阈值增加到: {game.min_collision_distance}")
        elif key == ord('-') or key == ord('_'):
            if game.min_collision_distance > 20:
                game.min_collision_distance -= 5
                print(f"碰撞阈值减少到: {game.min_collision_distance}")
        elif key == ord('f') or key == ord('F'):
            img_disp = game.put_chinese_text(img.copy(), "请输入食物图片路径(控制台):", (410, 400), font_size=32, color=(200, 0, 200))
            cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img_disp)
            cv2.waitKey(1)
            try:
                new_path = input("新食物图片路径: ").strip()
                if new_path:
                    game.set_food_image(new_path)
                    config_save()
            except Exception as e:
                print(f"图片更换异常: {e}")
        elif key == ord('h') or key == ord('H'):
            img_disp = game.put_chinese_text(img.copy(), "请输入左蛇头图片路径(控制台):", (410, 400), font_size=32, color=(255, 32, 255))
            cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img_disp)
            cv2.waitKey(1)
            try:
                new_path = input("左蛇头图片路径: ").strip()
                if new_path:
                    game.set_head_image("left", new_path)
                    config_save()
            except Exception as e:
                print(f"左蛇头更换异常: {e}")
        elif key == ord('j') or key == ord('J'):
            img_disp = game.put_chinese_text(img.copy(), "请输入右蛇头图片路径(控制台):", (410, 400), font_size=32, color=(32, 32, 255))
            cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img_disp)
            cv2.waitKey(1)
            try:
                new_path = input("右蛇头图片路径: ").strip()
                if new_path:
                    game.set_head_image("right", new_path)
                    config_save()
            except Exception as e:
                print(f"右蛇头更换异常: {e}")
        elif key == ord('b') or key == ord('B'):
            img_disp = game.put_chinese_text(img.copy(), "请输入新的背景音乐文件路径(控制台):", (410, 400), font_size=32, color=(90, 130, 255))
            cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img_disp)
            cv2.waitKey(1)
            try:
                new_bgm = input("新背景音乐文件路径: ").strip()
                if new_bgm:
                    switch_bgm(new_bgm)
                    globals()["BGM_FILE"] = new_bgm
                    config_save()
            except Exception as e:
                print(f"背景音乐更换异常: {e}")
        elif key == ord('c') or key == ord('C'):
            img_disp = game.put_chinese_text(img.copy(), "请输入左蛇吃食物音效文件路径(控制台):", (410, 400), font_size=32, color=(255, 120, 120))
            cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img_disp)
            cv2.waitKey(1)
            try:
                new_left = input("左蛇吃音效文件: ").strip()
                if new_left:
                    switch_left_eat_sound(new_left)
                    globals()["EAT_LEFT_FILE"] = new_left
                    config_save()
            except Exception as e:
                print(f"左蛇音效更换异常: {e}")
        elif key == ord('v') or key == ord('V'):
            img_disp = game.put_chinese_text(img.copy(), "请输入右蛇吃食物音效文件路径(控制台):", (410, 400), font_size=32, color=(150, 255, 150))
            cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img_disp)
            cv2.waitKey(1)
            try:
                new_right = input("右蛇吃音效文件: ").strip()
                if new_right:
                    switch_right_eat_sound(new_right)
                    globals()["EAT_RIGHT_FILE"] = new_right
                    config_save()
            except Exception as e:
                print(f"右蛇音效更换异常: {e}")
        elif key == ord('d') or key == ord('D'):
            img_disp = game.put_chinese_text(img.copy(), "请输入死亡音效文件路径(控制台):", (410, 400), font_size=32, color=(255, 90, 90))
            cv2.imshow("贪吃蛇(功能:图片/音效/音乐热切换+死亡音效)", img_disp)
            cv2.waitKey(1)
            try:
                new_dead = input("死亡音效文件: ").strip()
                if new_dead:
                    switch_dead_sound(new_dead)
                    globals()["DEAD_FILE"] = new_dead
                    config_save()
            except Exception as e:
                print(f"死亡音效更换异常: {e}")
except Exception as e:
    print("程序异常退出:", e)
finally:
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    cap.release()
    cv2.destroyAllWindows()
    gc.collect()
    config_save()