import customtkinter as ctk
import threading
import time
import os
import sys
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)
def run_core():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import math
    from pynput.mouse import Controller, Button
    import pynput.keyboard as pkey
    import pyautogui
    import time
    import tensorflow as tf
    import numpy as np
    import threading
    mouse = Controller()
   #Building model
    base_options = python.BaseOptions(model_asset_path=get_resource_path('hand_landmarker.task'))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2, 
        min_hand_detection_confidence=0.4, 
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    detector = vision.HandLandmarker.create_from_options(options)
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    prev_x, prev_y = 0, 0
    alpha = 0.1
    is_clicked = False
    is_scrolling = False
    anchor_y = 0
    DEADZONE = 35
    prev_scroll_y = 0
    prev_scroll_time = 0
    velocity = 0.0          
    FRICTION = 0.92       
    MIN_VELOCITY = 0.5  
    mouse_active = False        
    keyboard_active = False
    rock_start_time = None      
    board_start_time = None
    TOGGLE_DELAY = 0.8   
    last_toggle_time = 0  
    canvas = None
    prev_draw_x, prev_draw_y = 0, 0
    smooth_x, smooth_y = 0, 0       
    lost_frame_count = 0
    has_drawing = False            
    is_reading = False              
    smooth_l_dist = 100.0
    last_action_time = 0
    ACTION_COOLDOWN = 0.8 
    keyboard = pkey.Controller()
    try:
        model = tf.keras.models.load_model(get_resource_path('emnist_model.h5'))
    except:
        print("Download first!")
        exit()
    emnist_mapping = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
        19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
        28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
        45: 'r', 46: 't'
    }
    def is_finger(hand_landmarks, tip_idx, pip_idx):
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        wrist = hand_landmarks[0]
        dist_tip = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        dist_pip = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
        return dist_tip > dist_pip
    def process_character_in_background(image_to_read):
        nonlocal is_reading
        gray = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            char_img = thresh[y:y+h, x:x+w]
            size = max(w, h)
            pad_w, pad_h = (size - w) // 2, (size - h) // 2
            char_img = cv2.copyMakeBorder(char_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
            margin = int(size * 0.2)
            char_img = cv2.copyMakeBorder(char_img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
            char_img = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
            char_img = cv2.GaussianBlur(char_img, (3, 3), 0)
            char_img = char_img.astype('float32') / 255.0
            char_img = np.expand_dims(char_img, axis=(0, -1))
            prediction = model.predict(char_img, verbose=0)
            class_idx = np.argmax(prediction)
            result_char = emnist_mapping.get(class_idx, "?")
            print(f"Đã nhận diện chữ: {result_char}")
            keyboard.type(result_char)
        else:
            print("Bảng trắng trống!")
        is_reading = False

    def mapping(x, y, frame_width, frame_height, padding=150):
        x_min, x_max = padding, frame_width - padding
        y_min, y_max = padding, frame_height - padding
        x = max(x_min, min(x, x_max))
        y = max(y_min, min(y, y_max))
        screen_x = (x - x_min) / (x_max - x_min) * SCREEN_WIDTH
        screen_y = (y - y_min) / (y_max - y_min) * SCREEN_HEIGHT
        return int(screen_x), int(screen_y)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('MediaPipe Hand Tracking')
    
    cv2.setWindowProperty('MediaPipe Hand Tracking', cv2.WND_PROP_TOPMOST, 1)
    window_x = SCREEN_WIDTH - 640
    window_y = SCREEN_HEIGHT - 480 - 60  
    cv2.moveWindow('MediaPipe Hand Tracking', int(window_x), int(window_y))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        if canvas is None:
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        left_hand = None
        right_hand = None
        cursor_x, cursor_y = 0, 0
        left_hand_visible = False
        is_pinching_left = False

        if detection_result.hand_landmarks:
            for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    
                if handedness[0].category_name == "Left":
                    left_hand = hand_landmarks
                elif handedness[0].category_name == "Right":
                    right_hand = hand_landmarks
        current_time_sys = time.time()
        # ON mouse
        if left_hand:
            index_up = left_hand[8].y < left_hand[6].y
            pinky_up = left_hand[20].y < left_hand[18].y 
            middle_down = left_hand[12].y > left_hand[10].y 
            ring_down   = left_hand[16].y > left_hand[14].y 
            is_rock_left = index_up and pinky_up and middle_down and ring_down
            
            if is_rock_left and (current_time_sys - last_toggle_time > 1.5):
                if rock_start_time is None:
                    rock_start_time = time.time()  
                else:
                    elapsed = time.time() - rock_start_time
                    progress = int((elapsed / TOGGLE_DELAY) * 100)
                    cv2.putText(frame, f"MOUSE TOGGLE: {progress}%", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if elapsed >= TOGGLE_DELAY:
                        mouse_active = not mouse_active  
                        keyboard_active = False 
                        rock_start_time = None  
                        last_toggle_time = time.time() 
            else:
                rock_start_time = None
        else:
            rock_start_time = None
        # ON keyboard
        if right_hand:
            index_up = right_hand[8].y < right_hand[6].y
            pinky_up = right_hand[20].y < right_hand[18].y 
            middle_down = right_hand[12].y > right_hand[10].y 
            ring_down   = right_hand[16].y > right_hand[14].y 
            is_rock_right = index_up and pinky_up and middle_down and ring_down
            
            if is_rock_right and (current_time_sys - last_toggle_time > 1.5):
                if board_start_time is None:
                    board_start_time = time.time()  
                else:
                    elapsed = time.time() - board_start_time
                    progress = int((elapsed / TOGGLE_DELAY) * 100)
                    cv2.putText(frame, f"BOARD TOGGLE: {progress}%", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if elapsed >= TOGGLE_DELAY:
                        keyboard_active = not keyboard_active  
                        mouse_active = False 
                        board_start_time = None  
                        last_toggle_time = time.time() 
            else:
                board_start_time = None
        else:
            board_start_time = None

        is_switching_mode = (rock_start_time is not None) or (board_start_time is not None)
        # start keybaoard
        if keyboard_active and not is_switching_mode:
            if right_hand:
                r_thumb_x, r_thumb_y = int(right_hand[4].x * w), int(right_hand[4].y * h)
                r_index_x, r_index_y = int(right_hand[8].x * w), int(right_hand[8].y * h)
                ok_pinch_dist = math.sqrt((r_index_x - r_thumb_x)**2 + (r_index_y - r_thumb_y)**2)
                r_idx_ext = is_finger(right_hand, 8, 6)
                r_mid_ext = is_finger(right_hand, 12, 10)
                r_rng_ext = is_finger(right_hand, 16, 14)
                r_pnk_ext = is_finger(right_hand, 20, 18)
                thumb_tip_y = right_hand[4].y
                wrist_y = right_hand[0].y
                is_thumb_up = thumb_tip_y < (wrist_y - 0.05)
                is_thumb_down = thumb_tip_y > (wrist_y + 0.05)
                
                # function keyboard
                is_ok = (ok_pinch_dist < 40) and r_mid_ext and r_rng_ext and r_pnk_ext
                is_like = is_thumb_up and not r_idx_ext and not r_mid_ext and not r_rng_ext and not r_pnk_ext
                is_enter = is_thumb_up and r_idx_ext and not r_mid_ext and not r_rng_ext and not r_pnk_ext
                is_dislike = is_thumb_down and not r_idx_ext and not r_mid_ext and not r_rng_ext and not r_pnk_ext
                is_clear = r_idx_ext and r_mid_ext and r_rng_ext and r_pnk_ext # Xòe 5 ngón (High-five)
                
                current_time = time.time()
                if current_time - last_action_time > ACTION_COOLDOWN:
                    if is_ok and has_drawing and not is_reading:
                        is_reading = True
                        threading.Thread(target=process_character_in_background, args=(canvas.copy(),)).start()
                        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                        has_drawing = False
                        prev_draw_x, prev_draw_y = 0, 0 
                        last_action_time = current_time 
                    elif is_like:
                        keyboard.tap(pkey.Key.space)
                        print("Đã bấm Space!")
                        last_action_time = current_time                       
                    elif is_enter:
                        keyboard.tap(pkey.Key.enter)
                        print("Đã bấm Enter!")
                        last_action_time = current_time                   
                    elif is_clear:
                        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                        has_drawing = False
                        prev_draw_x, prev_draw_y = 0, 0
                        print("Đã xóa bảng trắng!")
                        last_action_time = current_time                    
                    elif is_dislike:
                        keyboard.tap(pkey.Key.backspace)
                        print("Đã xóa 1 ký tự (Backspace)!")
                        last_action_time = current_time
            if left_hand:
                left_hand_visible = True
                l_index_x, l_index_y = int(left_hand[8].x * w), int(left_hand[8].y * h)
                l_thumb_x, l_thumb_y = int(left_hand[4].x * w), int(left_hand[4].y * h)
                raw_l_dist = math.sqrt((l_index_x - l_thumb_x)**2 + (l_index_y - l_thumb_y)**2)
                if smooth_l_dist == 100.0:
                    smooth_l_dist = raw_l_dist
                else:
                    smooth_l_dist = 0.5 * raw_l_dist + 0.5 * smooth_l_dist
                    
                if smooth_l_dist < 25:
                    is_pinching_left = True 
                elif smooth_l_dist > 50:
                    is_pinching_left = False         
                if smooth_x == 0 and smooth_y == 0:
                    smooth_x, smooth_y = l_index_x, l_index_y
                else:
                    raw_jump = math.sqrt((l_index_x - smooth_x)**2 + (l_index_y - smooth_y)**2)
                    MAX_SPEED = 40.0
                    if raw_jump > MAX_SPEED:
                        ratio = MAX_SPEED / raw_jump
                        l_index_x = smooth_x + (l_index_x - smooth_x) * ratio
                        l_index_y = smooth_y + (l_index_y - smooth_y) * ratio
                    if raw_jump > 2:
                        smooth_x = int(0.4 * l_index_x + 0.6 * smooth_x)
                        smooth_y = int(0.4 * l_index_y + 0.6 * smooth_y)
                cursor_x, cursor_y = smooth_x, smooth_y    
                if is_pinching_left:
                    if prev_draw_x == 0 and prev_draw_y == 0:
                        prev_draw_x, prev_draw_y = smooth_x, smooth_y
                    cv2.line(canvas, (prev_draw_x, prev_draw_y), (smooth_x, smooth_y), (0, 0, 0), 20, cv2.LINE_AA)
                    prev_draw_x, prev_draw_y = smooth_x, smooth_y
                    has_drawing = True
                    lost_frame_count = 0
                else:
                    lost_frame_count += 1
                    if lost_frame_count > 2:
                        prev_draw_x, prev_draw_y = 0, 0      
            if not left_hand_visible:
                smooth_x, smooth_y = 0, 0
                prev_draw_x, prev_draw_y = 0, 0
                lost_frame_count = 0
                smooth_l_dist = 100.0
        #Start mouse
        if mouse_active and not is_switching_mode:
            if left_hand:
                x_click = (left_hand[8].x - left_hand[4].x) ** 2
                y_click = (left_hand[8].y - left_hand[4].y) ** 2
                click_dist = math.sqrt(x_click + y_click)
                x_frame = left_hand[8].x * w
                y_frame = left_hand[8].y * h
                target_x, target_y = mapping(x_frame, y_frame, w, h, padding=80)
                if prev_x == 0 and prev_y == 0:
                    smooth_x, smooth_y = target_x, target_y
                else:
                    raw_jump = math.sqrt((target_x - prev_x)**2 + (target_y - prev_y)**2)
                    MAX_SPEED = 80.0
                    if raw_jump > MAX_SPEED:
                        ratio = MAX_SPEED / raw_jump
                        target_x = prev_x + (target_x - prev_x) * ratio
                        target_y = prev_y + (target_y - prev_y) * ratio
                    adjusted_jump = math.sqrt((target_x - prev_x)**2 + (target_y - prev_y)**2)
                    #jitter -> EMA
                    if adjusted_jump > 3: 
                        smooth_x = 0.3 * target_x + 0.7 * prev_x
                        smooth_y = 0.3 * target_y + 0.7 * prev_y
                    else:
                        smooth_x, smooth_y = prev_x, prev_y 
                if smooth_x != 0 and smooth_y != 0:
                    mouse.position = (int(smooth_x), int(smooth_y))
                    prev_x, prev_y = smooth_x, smooth_y 
                #Clicking
                if click_dist < 0.03:
                    if not is_clicked:
                        mouse.click(Button.left, 1)
                        is_clicked = True 
                else:
                    is_clicked = False
            if right_hand:
                current_time = time.time()
                current_y = right_hand[8].y * h
                x_scroll = (right_hand[4].x - right_hand[8].x) ** 2
                y_scroll = (right_hand[4].y - right_hand[8].y) ** 2
                scale_croll = math.sqrt(x_scroll + y_scroll)
                if scale_croll < 0.04:
                    if not is_scrolling:
                        is_scrolling = True
                        prev_scroll_y = current_y
                        prev_scroll_time = current_time
                        velocity = 0.0
                    else:
                        dt = current_time - prev_scroll_time
                        if dt > 0:
                            delta_y = current_y - prev_scroll_y
                            inst_velocity = delta_y / dt 
                            velocity = 0.6 * velocity + 0.4 * inst_velocity
                            if abs(delta_y) > 2:
                                mouse.scroll(0, int(delta_y / 8))
                            prev_scroll_y = current_y
                            prev_scroll_time = current_time
                else:
                    is_scrolling = False
                    if abs(velocity) > MIN_VELOCITY:
                        scroll_step = velocity * 0.005  
                        if abs(scroll_step) >= 1:
                            mouse.scroll(0, int(-scroll_step))
                        velocity *= FRICTION
                    else:
                        velocity = 0.0     
        elif is_switching_mode:
            cv2.putText(frame, "FREEZING SYSTEM...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if keyboard_active:
            cv2.putText(frame, "MODE: DRAWING", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            ink_mask = (canvas[:, :, 0] == 0) & (canvas[:, :, 1] == 0) & (canvas[:, :, 2] == 0)
            frame[ink_mask] = [0, 0, 255] 
            if left_hand_visible:
                if is_pinching_left:
                    cv2.circle(frame, (cursor_x, cursor_y), 8, (0, 0, 255), -1) 
                else:
                    cv2.circle(frame, (cursor_x, cursor_y), 5, (255, 0, 0), 2)  
        elif mouse_active:
            cv2.putText(frame, "MODE: MOUSE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "MODE: IDLE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 3)
        cv2.imshow('MediaPipe Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    app.after(0, reset_start_button)
#gui
def start_btn_event():
    start_btn.configure(state="disabled", text="[ SYSTEM RUNNING ]", fg_color="#006600")
    ai_thread = threading.Thread(target=run_core, daemon=True)
    ai_thread.start()
def reset_start_button():
    start_btn.configure(state="normal", text="START INITIALIZATION", fg_color="#1f538d")
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue") 
app = ctk.CTk()
app.iconbitmap(get_resource_path('projecting-future-hand-holding-holographic-projection-38996155.ico'))
app.geometry("400x550")
app.title("OBHS")
app.configure(fg_color="#0a0a0a") 
main_frame = ctk.CTkFrame(master=app, fg_color="#121212", border_width=1, border_color="#333333", corner_radius=15)
main_frame.pack(pady=40, padx=40, fill="both", expand=True)
title_label = ctk.CTkLabel(
    master=main_frame, 
    text="ONLY BY HANDS", 
    font=ctk.CTkFont(family="Courier New", size=28, weight="bold"),
    text_color="#00FFAA" 
)
title_label.pack(pady=(40, 10))
subtitle_label2 = ctk.CTkLabel(
    master=main_frame, 
    text="Just with hands. No mouse, no keyboard.", 
    font=ctk.CTkFont(family="Courier New", size=13),
    text_color="#555555"
)
subtitle_label2.pack(pady=(0, 40))
subtitle_label1 = ctk.CTkLabel(
    master=main_frame, 
    text="Demo version", 
    font=ctk.CTkFont(family="Courier New", size=14),
    text_color="#555555"
)
subtitle_label1.pack(pady=(0, 20))
start_btn = ctk.CTkButton(
    master=main_frame,
    text="START INITIALIZATION",
    font=ctk.CTkFont(family="Courier New", size=14, weight="bold"),
    height=45,
    corner_radius=4,
    command=start_btn_event
)
start_btn.pack(pady=15, padx=20, fill="x")

settings_btn = ctk.CTkButton(
    master=main_frame,
    text="SYSTEM SETTINGS",
    font=ctk.CTkFont(family="Courier New", size=14),
    height=45,
    corner_radius=4,
    fg_color="#1a1a1a",
    text_color="#555555",
    hover=False,
    state="disabled"
)
settings_btn.pack(pady=15, padx=20, fill="x")
guide_btn = ctk.CTkButton(
    master=main_frame,
    text="USER GUIDE DOCS",
    font=ctk.CTkFont(family="Courier New", size=14),
    height=45,
    corner_radius=4,
    fg_color="#1a1a1a",
    text_color="#555555",
    hover=False,
    state="disabled"
)
guide_btn.pack(pady=15, padx=20, fill="x")
footer_label = ctk.CTkLabel(
    master=app, 
    text="[ SECURE CONNECTION ESTABLISHED ]", 
    font=ctk.CTkFont(family="Courier New", size=10),
    text_color="#333333"
)
footer_label.pack(side="bottom", pady=15)
app.mainloop()