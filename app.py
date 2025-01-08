import streamlit as st

st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")

# 在側邊欄建立目錄
menu = st.sidebar.selectbox(
    "目錄",
    ["首頁","摘要", "第一章 智慧機械設計概論", "第二章 ROS/AMR智慧功能設計", "第三章 結果與討論", "第四章 結論與心得", "參考文獻", "分工表"]
)

# 根據選項顯示內容
if menu == "首頁":
    st.title("113(上)學年度『智慧機械設計』課程期末報告")

    st.title("ROS自主移動平台與AI整合之研究")

    st.header("指導老師：周榮源")
    st.header("班級：碩設計一甲")
    st.header("組別：第一組")
    st.header("組員：11373106/陳彥碩、11373122/林群祐、11373137/廖翊翔")

    st.write("歡迎來到智慧機械設計課程期末報告。請從左側選單選擇要查看的實驗項目。")

    # 不需要在這裡定義實驗項目列表，因為Streamlit會自動生成側邊欄

    # 顯示一張圖片(image)
    st.image("Pictures + Videos/1.jpg")
    # Caption
    st.caption("""Turtlebot3""")


elif menu == "摘要":
    st.title("摘要")
    st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本學期的智慧機械設計課程，
                主要學習如何透過ROS操控機器人、
                機器學習與影像辨識、分割等等的智慧機械設計實作，
                也學習到了如何整合這些技術到同一台機器人身上以及
                透過Streamlit設計網頁來將設計資料可視化。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本報告首先將會介紹智慧機械概論與其產業應用以及相關技術的發展現況、
                各領域之應用狀況等等，並將上述智慧機械設計實作相關技術分為五項介紹，
                分別是Turtlebot3(Burger)之避障與導航實作、ROS/AMR之深度學習影像移動控制功能實作、
                ultralytics YOLO與SAM物件偵測與分割功能實作、three-link planar manipulator模擬實作與
                streamlit UI設計與資料可視化，內文會詳細介紹每個研究項目的研究方法與步驟，並在最後附上研究成果與討論。
                </p>
                """, unsafe_allow_html=True)

    
elif menu == "第一章 智慧機械設計概論":
    
    st.title("1.1 智慧機械概論")
    st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                智慧機械是整合機械工程、物聯網、人工智慧與自動化技術等跨領域的產物，
                隨著科技的進步以及近年來AI的大幅發展，提升機械設備的自主性、靈活性與效率已是未來的趨勢，
                尤其智慧機械在製造業的發展更是令人期待，在這個大缺工的時代，若智慧機械可以取代大量人力，
                對於以代工為主的我國來說，必定是缺工問題的解方，且智慧機械可以消除「人」在各方面的不穩定性，
                進一步的提升產量、良率，進而壓低成本，同時又維持穩定的品質，想必此解方有助於我國應對中國與東南亞國家大量廉價勞動力的優勢。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                智慧機械的實現需依賴多項核心技術，包括：感測與數據收集技術、人工智慧與機器學習、自動控制技術和物聯網連接等，
                透過嵌入式感測器、相機及雷達等設備，智慧機械能夠實時獲取環境及操作數據，再來可以利用AI算法分析數據，
                預測潛在問題並作出最佳決策，基於收集的數據和AI分析結果，讓智慧機械可以自主調整其運行狀態，
                最後智慧機械透過物聯網實現與其他設備的互聯，形成高效協同的智能生態系統。
                </p>
                """, unsafe_allow_html=True)

    st.title("1.2 智慧機械之產業應用")
    st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                智慧機械在多個領域中皆發揮著重要作用，以製造業來說，自動化生產線、機器人焊接及智慧工廠的運行都須依賴智慧機械的支持。
                此外，醫療領域可以應用醫療機器人協助完成精密手術或使用智慧檢測設備則提升診斷準確性。
                在農業領域同樣可以應用智慧農機來根據環境數據自動調整灑水、施肥及收割方案。最後在日常生活中，
                智慧家電、無人駕駛車輛及物流機器人已成為日益普及的應用。 
                </p>
                """, unsafe_allow_html=True)
    

elif menu == "第二章 ROS/AMR智慧功能設計":
    
    submenu = st.sidebar.radio("關於選單", ["2.1 Turtlebot3(Burger)之避障與導航實作", "2.2 ROS/AMR之深度學習影像移動控制功能實作", "2.3 ultralytics YOLO與SAM物件偵測與分割功能實作", "2.4 three-link planar manipulator模擬實作", "2.5 streamlit UI設計與資料可視化"])
    if submenu == "2.1 Turtlebot3(Burger)之避障與導航實作":
        st.title('2.1 Turtlebot3(Burger)之避障與導航實作')
        st.markdown('### 步驟1：啟動VirtualBox→開啟設定→顯示→啟用3D加速')
        st.image("Pictures + Videos/ROS_navigation/2.png")

        st.markdown('### 步驟2：網路→選取橋接介面卡')
        st.image("Pictures + Videos/ROS_navigation/3.png")

        st.markdown('### 步驟3：創建資料夾(catkin_ws)，並在裡面建立src資料夾')
        st.image("Pictures + Videos/ROS_navigation/4.png")

        st.markdown('### 步驟4：打開Terminal→輸入以下程式(下載檔案到src)')
        code = """
        git clone https://github.com/ROBOTIS-GIT/turtlebot3
        """
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/6.png")

        st.markdown('### 步驟5：退回catkin_ws，並catkin_make')
        code = """
        cd ..
        catkin_make
        """
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/7.png")

        st.markdown('### 步驟6：輸入以下程式')
        code = """
        roscore
        """ 
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/8.png")

        st.markdown('### 步驟7：跟Turtlebt3連線(192.168.1.199為實驗室網路IP)、password：raspberry、最後輸入launch檔')
        code = """
        ssh pi@192.168.1.199
        raspberry
        roslaunch turtlebot3_bringup turtlebot3_robot.launch
        """ 
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/9.png")

        st.markdown('### 步驟8：匯入burger模型→輸入slam的launch檔')
        code = """
        export TURTLEBOT3_MODEL=burger 
        roslaunch turtlebot3_slam turtlebot3_slam.launch 
        """ 
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/10.png")

        st.markdown('### 步驟9：啟動地圖')
        st.image("Pictures + Videos/ROS_navigation/11.png")

        st.markdown('### 步驟10：匯入burger模型→輸入keyboard的launch檔(用鍵盤控制Turtlebot3)')
        code = """
        export TURTLEBOT3_MODEL=burger 
        roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch 
        """ 
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/12.png")

        st.markdown('### 步驟11：掃描地形')
        st.video("Pictures + Videos/ROS_navigation/13.mp4")
        st.video("Pictures + Videos/ROS_navigation/try.mp4")

        st.markdown('### 步驟12：完成地形的掃描後，儲存地圖')
        code = """
        rosrun map_server map_saver -f ~/map
        """ 
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/14.png")

        st.markdown('### 步驟13：開啟地圖')
        code = """
        roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml
        """ 
        st.code(code, language="python")
        st.image("Pictures + Videos/ROS_navigation/15.png")

        st.markdown('### 步驟14：開始導航與避障測試')
        st.video("Pictures + Videos/ROS_navigation/final.mp4")


    elif submenu == "2.2 ROS/AMR之深度學習影像移動控制功能實作":
            st.title('2.2 ROS/AMR之深度學習影像移動控制功能實作')
            st.markdown('### 步驟1：撰寫一個由yolo11n訓練的模型，訓練的目標為辨識左轉、右轉、前進、後退與停止的符號')
            code = """
        import os
        from ultralytics import YOLO
        import cv2

        # 1. 自動生成 YAML 文件
        def create_yaml_file(dataset_path, yaml_path):
            yaml_content = f\""" 
        path: {dataset_path}
        train: {os.path.join(dataset_path, 'images/train')}
        val: {os.path.join(dataset_path, 'images/val')}

        nc: 5  # 類別數量（左右符號，共 2 個類別）
        names: ['Go straight', 'Left', 'Right', 'Start', 'Stop']  # 類別名稱
        \""" 
            with open(yaml_path, "w") as file:
                file.write(yaml_content)
            print(f"YAML 文件已創建：{yaml_path}")

        # 2. 訓練 YOLO 模型
        def train_yolo_model(yaml_path, model_name="yolo11n.pt", epochs=50, imgsz=640):
            print("開始訓練 YOLO 模型...")
            model = YOLO(model_name)
            model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                name="left_right_detection",
                device='0'  # 使用 GPU
            )
            print("模型訓練完成！最佳權重文件保存在：runs/detect/left_right_detection/weights/best.pt")

        # 3. 即時檢測
        def detect_with_webcam(weight_path, device_index=0):
            print("正在使用攝影機進行即時檢測...")
            model = YOLO(weight_path)
            cap = cv2.VideoCapture(device_index)

            if not cap.isOpened():
                print("無法打開攝影機！請檢查設備編號。")
                return

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("無法讀取攝影機畫面")
                    break

                # 使用模型進行預測
                results = model(frame)

                # 繪製檢測結果
                annotated_frame = results[0].plot()
                cv2.imshow("YOLOv8 Left-Right Detection", annotated_frame)

                # 按 'q' 鍵退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        # 主函數
        if __name__ == "__main__":
            # 配置路徑
            dataset_path = "C:/Users/USER/OneDrive/桌面/llm_streamlit/YOLO/datasets"  # 數據集的根目錄
            yaml_path = "LR.yaml"  # YAML 文件路徑
            weight_path = "runs/detect/left_right_detection/weights/best.pt"  # 最佳權重文件

            # 自動創建 YAML 文件
            if not os.path.exists(yaml_path):
                create_yaml_file(dataset_path, yaml_path)

            # 檢查是否已訓練模型
            if not os.path.exists(weight_path):
                # 訓練模型
                train_yolo_model(yaml_path)

            # 即時檢測
            detect_with_webcam(weight_path, device_index=1)  # 修改設備編號為 1 以使用外接攝影機
        """
            with st.expander("點擊展開完整程式碼"):
                 st.code(code, language="python")


            st.markdown('### 步驟2：將Turtlebot3外掛一個鏡頭並與Turtlebot3連接，最後與步驟1訓練好的最佳權重檔整合並設定馬達轉數等參數')     
            code = """
        from ultralytics import YOLO
        import cv2
        import rospy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import CompressedImage
        from cv_bridge import CvBridge, CvBridgeError
        import time

        class YOLOGestureControl:
            def __init__(self, model_path):
                # Load YOLO model
                self.model = YOLO(model_path)
                rospy.init_node('yolo_gesture_control', anonymous=True)
                self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
                self.bridge = CvBridge()
                self.frame = None

                # Subscribe to ROS image topic
                rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.image_callback)

            def image_callback(self, msg):
                try:
                    # Convert ROS message to OpenCV format using cv_bridge
                    self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                except CvBridgeError as e:
                    rospy.logerr(f"Failed to convert image: {e}")

            def detect_gesture(self):
                print("Press Ctrl+C to exit the program")

                while not rospy.is_shutdown():
                    if self.frame is None:
                        rospy.logwarn("No frame received yet")
                        rospy.sleep(0.1)
                        continue

                    # Detect using YOLO model
                    results = self.model.predict(self.frame)

                    # Annotate detection results
                    annotated_frame = results[0].plot()

                    # Determine detected object
                    detected_label = None
                    for box in results[0].boxes:
                        detected_label = self.model.names[int(box.cls[0])]  # Get object name
                        break  # Only process the first detected object

                    if detected_label:
                        self.handle_detection(detected_label)  # Keep label as-is for case-sensitive comparison

                    # Display results
                    cv2.imshow("YOLO Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()

            def handle_detection(self, label):
                rospy.loginfo(f"Detected label: {label}")  # Debug output

                if label == 'Go straight':
                    rospy.loginfo("Triggering forward motion.")  # Debug output
                    self.send_command('forward')
                elif label in ['Left', 'Right', 'Stop']:
                    rospy.loginfo(f"Handling action: {label}")
                    self.send_command(label)

            def send_command(self, action):
                twist = Twist()
                rospy.loginfo(f"Sending command: {action}")

                if action == 'Left':
                    twist.angular.z = 1.57  # Left turn
                    self.pub.publish(twist)
                    rospy.sleep(1)
                    twist.angular.z = 0.0
                    self.pub.publish(twist)
                elif action == 'Right':
                    twist.angular.z = -1.57  # Right turn
                    self.pub.publish(twist)
                    rospy.sleep(1)
                    twist.angular.z = 0.0
                    self.pub.publish(twist)
                elif action == 'forward':
                    twist.linear.x = 0.1  # Move forward with speed 0.2
                    self.pub.publish(twist)
                    rospy.sleep(3)
                    twist.linear.x = 0.0
                    self.pub.publish(twist)
                elif action == 'Start':
                    twist.linear.x = -0.2
                    pub.publish(twist)
                    rospy.sleep(3)  # 倒車3秒
                    twist.linear.x = 0.0
                    pub.publish(twist)  # 停止車輛

                elif action == 'Stop':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0  # Stop moving
                    self.pub.publish(twist)

                rospy.loginfo(f"Published Twist: linear.x = {twist.linear.x}, angular.z = {twist.angular.z}")

        if __name__ == "__main__":
            try:
                model_path = "/media/sf_YOLO/runs/detect/left_right_detection/weights/best.pt"  # Replace with your model path
                yolo_gesture_control = YOLOGestureControl(model_path)
                yolo_gesture_control.detect_gesture()
            except rospy.ROSInterruptException:
                pass

        """
            with st.expander("點擊展開完整程式碼"):
                st.code(code, language="python")
                
            st.markdown('### 步驟3：啟動VirtualBox→開啟設定→共用資料夾→新增資料夾路徑(步驟2的資料夾路徑)')
            st.image("Pictures + Videos/ROS_navigation/1-11.png")

            st.markdown('### 步驟4：重複2-1Turtlebot3(Burger)之避障與導航實作的步驟1~7')

            st.markdown('### 步驟5：cd到步驟3新增的資料夾→輸入步驟2建立的py檔')
            st.image("Pictures + Videos/ROS_navigation/1-12.png")  

            st.markdown('### 步驟6：測試功能')
            st.video("Pictures + Videos/YOLO+ROS_Turtlebot/try.mp4")


    elif submenu == "2.3 ultralytics YOLO與SAM物件偵測與分割功能實作":
            st.title('2.3 ultralytics YOLO與SAM物件偵測與分割功能實作')
            st.markdown('### 步驟1：確認要辨認的物體，本題目依題目設定目標物體為杏鮑菇、香蕉、芭樂')

            st.markdown('### 步驟2：三個物體各尋找150~200張照片')

            st.markdown('### 步驟3：標註圖片，可使用線上標註網站"[roboflow](https://app.roboflow.com/test-6swly)"或安裝labelImg套件')
            st.markdown('##### 步驟3_方法1：roboflow標註杏鮑菇')
            st.image("Pictures + Videos/YOLO+SAM/k_2.png")
            code = """
            #步驟3_方法2：安裝 LabelImg
            pip install labelImg

            # 執行 LabelImg
            labelImg
            """
            st.code(code, language="python")
            st.image("Pictures + Videos/YOLO+SAM/b.png")

            st.markdown('### 步驟4：撰寫一個由yolo11n訓練的模型，訓練的目標為辨識杏鮑菇、香蕉與芭樂的')
            code = """
            import os
            from ultralytics import YOLO
            import cv2

            # 1. 自動生成 YAML 文件
            def create_yaml_file(dataset_path, yaml_path):
                yaml_content = f\""" 
            path: {dataset_path}
            train: {os.path.join(dataset_path, 'images/train_banana')}
            val: {os.path.join(dataset_path, 'images/val_banana')}

            nc: 3  # 類別數量（左右符號，共 5 個類別）
            names: ['Banana', 'Guava', 'King Oyster Mushroom']  # 類別名稱
            \"""
                with open(yaml_path, "w") as file:
                    file.write(yaml_content)
                print(f"YAML 文件已創建：{yaml_path}")

            # 2. 訓練 YOLO 模型
            def train_yolo_model(yaml_path, model_name="yolo11n.pt", epochs=50, imgsz=640):
                print("開始訓練 YOLO 模型...")
                model = YOLO(model_name)
                model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    name="King Oyster Mushroom",
                    device='0'  # 使用 GPU
                )
                print("模型訓練完成！最佳權重文件保存在：runs/detect/King Oyster Mushroom/weights/best.pt")

            # 3. 實時檢測
            def detect_with_webcam(weight_path, device_index=0):
                print("正在使用攝影機進行實時檢測...")
                model = YOLO(weight_path)
                cap = cv2.VideoCapture(device_index)

                if not cap.isOpened():
                    print("無法打開攝影機！請檢查設備編號。")
                    return

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("無法讀取攝影機畫面")
                        break

                    # 使用模型進行預測
                    results = model(frame)

                    # 繪製檢測結果
                    annotated_frame = results[0].plot()
                    cv2.imshow("YOLOv8 Left-Right Detection", annotated_frame)

                    # 按 'q' 鍵退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

            # 主函數
            if __name__ == "__main__":
                # 配置路徑
                dataset_path = "C:/Users/USER/OneDrive/桌面/llm_streamlit/YOLO/datasets"  # 數據集的根目錄
                yaml_path = "custom_dataset.yaml"  # YAML 文件路徑
                weight_path = "runs/detect/King Oyster Mushroom/weights/best.pt"  # 最佳權重文件

                # 自動創建 YAML 文件
                if not os.path.exists(yaml_path):
                    create_yaml_file(dataset_path, yaml_path)

                # 檢查是否已訓練模型
                if not os.path.exists(weight_path):
                    # 訓練模型
                    train_yolo_model(yaml_path)

                # 實時檢測
                detect_with_webcam(weight_path, device_index=1)  # 修改設備編號為 1 以使用外接攝影機
            """
            with st.expander("點擊展開完整程式碼"):
                st.code(code, language="python")
            
            st.markdown('### 步驟5：確認是否能辨認物體')
            st.image("Pictures + Videos/YOLO+SAM/k.png")
            st.image("Pictures + Videos/YOLO+SAM/g.png")

            st.markdown('### 步驟6：在SAM模型中整合步驟4訓練好的最佳權重檔，並設定使用GPU計算')
            code = """
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

# 設備選擇，優先使用 GPU
def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用 GPU 進行推理")
    else:
        device = torch.device('cpu')
        print("使用 CPU 進行推理")
    return device

def main():
    # 初始化設備
    device = select_device()

    # 加載自定義訓練的 YOLO 模型權重
    model_path = "runs/detect/King Oyster Mushroom/weights/best.pt"
    yolo_model = YOLO(model_path).to(device)

    # 加載 SAM 模型
    sam_checkpoint = "C:/Users/USER/OneDrive/桌面/llm_streamlit/YOLO/segment-anything/sam_vit_b_01ec64.pth"  # 替換為您的 SAM 權重文件路徑
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    # 打開攝影機
    cap = cv2.VideoCapture(1)  # 修改為 0 或其他攝影機索引

    if not cap.isOpened():
        print("無法打開攝影機！")
        return

    print("按 'q' 鍵退出檢測")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面")
            break

        # 使用 YOLO 模型進行推理
        results = yolo_model(frame, conf=0.5, device=device)

        # 獲取目標檢測框，並使用 SAM 進行分割
        for result in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, result[:4])
            cropped_frame = frame[y1:y2, x1:x2].copy()

            if cropped_frame.size == 0:
                continue  # 跳過無效裁剪區域

            # 使用 SAM 模型
            predictor.set_image(frame)  # 設置整張圖片
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False
            )

            # 在畫面上繪製分割結果
            if masks is not None and len(masks) > 0:
                mask = masks[0]
                frame[mask > 0] = [0, 255, 0]  # 用綠色標記分割區域

        # 可視化檢測結果
        annotated_frame = results[0].plot()

        # 顯示檢測結果
        cv2.imshow("YOLOv8 + SAM Detection", annotated_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

            """
            with st.expander("點擊展開完整程式碼"):
                st.code(code, language="python")

            st.markdown('### 步驟7：確認是否能分割物體')
            st.image("Pictures + Videos/YOLO+SAM/SMA_K (2).png")
 

    elif submenu == "2.4 three-link planar manipulator模擬實作":
            st.title('2.4 three-link planar manipulator模擬實作')
            st.markdown('## (a)ROS moveit')
            st.markdown('### 步驟1：創建workspace資料夾→內建src資料夾→git clone https://github.com/Robotawi/rrr-arm.git')
            code = """
            mkdir workspace  # 創建workspace資料夾

            cd workspace     # 進入workspace資料夾

            mkdir src        # 創建src資料夾

            cd src           # 進入src資料夾

            git clone https://github.com/Robotawi/rrr-arm.git  #下載github檔案
            """
            st.code(code, language="python") 
            st.image("Pictures + Videos/threelink/1.png")

            st.markdown('### 步驟2：cd回workspace資料夾→重新編譯並儲存')
            code = """
            catkin_make  # 重新編譯

            source devel/setup.bash  # 儲存
            """
            st.code(code, language="python") 
            st.image("Pictures + Videos/threelink/3.png")

            st.markdown('### 步驟3：進入src，啟動view_arm.launch檔→跳出手臂畫面')
            code = """
            roslaunch rrr_arm view_arm.launch  # 啟動launch檔
            """
            st.code(code, language="python")
            st.image("Pictures + Videos/threelink/4.png")
            st.image("Pictures + Videos/threelink/5.png")

            st.markdown('### 步驟4：確認沒問題即退出手臂畫面，並開啟view_arm_gazebo_control_empty_world.launch檔')
            code = """
            roslaunch rrr_arm view_arm_gazebo_control_empty_world.launch   # 啟動launch檔
            """
            st.code(code, language="python")
            st.image("Pictures + Videos/threelink/6.png")

            st.markdown('### 步驟5：在新Terminal輸入rostopic list，確認是否有以下內容')
            st.image("Pictures + Videos/threelink/7.png")

            st.markdown('### 步驟6：輸入以下程式以測試手臂→結束')
            code = """
            rostopic pub /rrr_arm/joint1_position_controller/command  std_msgs/Float64 "data: 1.0" & rostopic pub /rrr_arm/joint2_position_controller/command  std_msgs/Float64 "data: 1.0" & rostopic pub /rrr_arm/joint3_position_controller/command  std_msgs/Float64 "data: 1.5" & rostopic pub /rrr_arm/joint4_position_controller/command std_msgs/Float64 "data: 1.5"   # 移動手臂
            """
            st.code(code, language="python")
            st.image("Pictures + Videos/threelink/9.png")
            


            st.markdown('## (b)RL強化學習訓練')
            st.markdown('### 步驟1：創建env.py，開始設定環境參數，如手臂尺寸、正向運動學方程式等等')
            code = """
            import numpy as np
            import pyglet

            class ArmEnv(object):
                viewer = None
                dt = .1    # refresh rate
                action_bound = [-1, 1]
                goal = {'x': 150., 'y': 150., 'l': 40}
                state_dim = 13  # 狀態維度擴展
                action_dim = 3  # 動作維度擴展

                def __init__(self):
                    self.arm_info = np.zeros(
                        3, dtype=[('l', np.float32), ('r', np.float32)])
                    self.arm_info['l'] = [100, 100, 100]  # 三連桿長度
                    self.arm_info['r'] = [np.pi / 6] * 3  # 三個角度初始化
                    self.on_goal = 0

                def step(self, action):
                    done = False
                    action = np.clip(action, *self.action_bound)
                    self.arm_info['r'] += action * self.dt
                    self.arm_info['r'] %= np.pi * 2  # 角度歸一化到 [0, 2π]

                    # 正向運動學計算
                    (l1, l2, l3) = self.arm_info['l']
                    (r1, r2, r3) = self.arm_info['r']
                    a1 = np.array([200., 200.])
                    a1_ = a1 + np.array([np.cos(r1), np.sin(r1)]) * l1
                    a2_ = a1_ + np.array([np.cos(r1 + r2), np.sin(r1 + r2)]) * l2
                    finger = a2_ + np.array([np.cos(r1 + r2 + r3), np.sin(r1 + r2 + r3)]) * l3

                    # 計算獎勵和是否到達目標
                    dist1 = [(self.goal['x'] - a1_[0]) / 400, (self.goal['y'] - a1_[1]) / 400]
                    dist2 = [(self.goal['x'] - a2_[0]) / 400, (self.goal['y'] - a2_[1]) / 400]
                    dist3 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
                    r = -np.sqrt(dist3[0]**2 + dist3[1]**2)

                    if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2 and \
                    self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                        r += 1.
                        self.on_goal += 1
                        if self.on_goal > 50:
                            done = True
                    else:
                        self.on_goal = 0

                    s = np.concatenate((a1_/200, a2_/200, finger/200, dist1, dist2, dist3, [1. if self.on_goal else 0.]))
                    return s, r, done

                def reset(self):
                    self.goal['x'] = np.random.rand() * 400
                    self.goal['y'] = np.random.rand() * 400
                    self.arm_info['r'] = 2 * np.pi * np.random.rand(3)
                    self.on_goal = 0
                    return self.step(np.zeros(3))[0]

                def render(self):
                    if self.viewer is None:
                        self.viewer = Viewer(self.arm_info, self.goal)
                    self.viewer.render()

                def sample_action(self):
                    return np.random.rand(3) - 0.5  # 三個動作

            class Viewer(pyglet.window.Window):
                bar_thc = 5

                def __init__(self, arm_info, goal):
                    super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
                    pyglet.gl.glClearColor(1, 1, 1, 1)
                    self.arm_info = arm_info
                    self.goal_info = goal
                    self.center_coord = np.array([200, 200])

                    self.batch = pyglet.graphics.Batch()

                    # 設定目標區域
                    goal_x, goal_y, goal_l = goal['x'], goal['y'], goal['l']
                    self.goal = pyglet.shapes.Rectangle(goal_x - goal_l / 2, goal_y - goal_l / 2, 
                                                        goal_l, goal_l, color=(86, 109, 249), batch=self.batch)

                    # 設定三連桿
                    self.arm1 = pyglet.shapes.Line(0, 0, 0, 0, width=self.bar_thc, color=(249, 86, 86), batch=self.batch)
                    self.arm2 = pyglet.shapes.Line(0, 0, 0, 0, width=self.bar_thc, color=(249, 86, 86), batch=self.batch)
                    self.arm3 = pyglet.shapes.Line(0, 0, 0, 0, width=self.bar_thc, color=(249, 86, 86), batch=self.batch)

                def render(self):
                    self._update_arm()
                    self.switch_to()
                    self.dispatch_events()
                    self.dispatch_event('on_draw')
                    self.flip()

                def on_draw(self):
                    self.clear()
                    self.batch.draw()

                def _update_arm(self):
                    (l1, l2, l3) = self.arm_info['l']
                    (r1, r2, r3) = self.arm_info['r']
                    a1 = self.center_coord
                    a1_ = a1 + np.array([np.cos(r1), np.sin(r1)]) * l1
                    a2_ = a1_ + np.array([np.cos(r1 + r2), np.sin(r1 + r2)]) * l2
                    a3_ = a2_ + np.array([np.cos(r1 + r2 + r3), np.sin(r1 + r2 + r3)]) * l3

                # 更新每段連桿的座標
                    self.arm1.x, self.arm1.y = a1[0], a1[1]
                    self.arm1.x2, self.arm1.y2 = a1_[0], a1_[1]

                    self.arm2.x, self.arm2.y = a1_[0], a1_[1]
                    self.arm2.x2, self.arm2.y2 = a2_[0], a2_[1]

                    self.arm3.x, self.arm3.y = a2_[0], a2_[1]
                    self.arm3.x2, self.arm3.y2 = a3_[0], a3_[1]


                def on_mouse_motion(self, x, y, dx, dy):
                    self.goal_info['x'] = x
                    self.goal_info['y'] = y
            """
            with st.expander("點擊展開完整程式碼"):
                st.code(code, language="python")


            st.markdown('### 步驟2：創建rl.py，利用tensorflow開始訓練手臂抓取方塊並設定權重檔儲存路徑')
            code = """
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers # type: ignore
            import os  # 新增 os 模組用於目錄操作

            class DDPG(object):
                def __init__(self, state_dim, action_dim, action_bound, lr_a=0.001, lr_c=0.002, gamma=0.9, tau=0.01, memory_capacity=10000, batch_size=64):
                    self.state_dim = state_dim
                    self.action_dim = action_dim
                    self.action_bound = action_bound
                    self.lr_a = lr_a
                    self.lr_c = lr_c
                    self.gamma = gamma
                    self.tau = tau
                    self.memory_capacity = memory_capacity
                    self.batch_size = batch_size
                    self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1), dtype=np.float32)
                    self.pointer = 0

                    # Build Actor and Critic networks
                    self.actor = self._build_actor()
                    self.target_actor = self._build_actor()
                    self.critic = self._build_critic()
                    self.target_critic = self._build_critic()

                    # Sync target networks with main networks
                    self._update_target(self.target_actor, self.actor, tau=1)
                    self._update_target(self.target_critic, self.critic, tau=1)

                    self.actor_optimizer = optimizers.Adam(learning_rate=self.lr_a)
                    self.critic_optimizer = optimizers.Adam(learning_rate=self.lr_c)

                def _build_actor(self):
                    model = models.Sequential([
                        layers.Input(shape=(self.state_dim,)),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(self.action_dim, activation='tanh'),
                        layers.Lambda(lambda x: x * tf.convert_to_tensor(self.action_bound, dtype=tf.float32))  # 保證 action_bound 是三維
                    ])
                    return model

                def _build_critic(self):
                    state_input = layers.Input(shape=(self.state_dim,))
                    action_input = layers.Input(shape=(self.action_dim,))
                    concat = layers.Concatenate()([state_input, action_input])

                    out = layers.Dense(30, activation='relu')(concat)
                    out = layers.Dense(1)(out)
                    return models.Model([state_input, action_input], out)

                def _update_target(self, target_model, model, tau):
                    for target_weights, weights in zip(target_model.weights, model.weights):
                        target_weights.assign(tau * weights + (1 - tau) * target_weights)

                def choose_action(self, state):
                    state = state[np.newaxis, :]
                    return self.actor.predict(state)[0]

                def learn(self):
                    indices = np.random.choice(self.memory_capacity, size=self.batch_size)
                    bt = self.memory[indices, :]
                    bs = bt[:, :self.state_dim]
                    ba = bt[:, self.state_dim: self.state_dim + self.action_dim]
                    br = bt[:, -self.state_dim - 1: -self.state_dim]
                    bs_ = bt[:, -self.state_dim:]

                    # Update Critic
                    with tf.GradientTape() as tape:
                        target_actions = self.target_actor(bs_)
                        y = br + self.gamma * self.target_critic([bs_, target_actions])
                        q = self.critic([bs, ba])
                        critic_loss = tf.reduce_mean(tf.square(y - q))
                    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                    # Update Actor
                    with tf.GradientTape() as tape:
                        actions = self.actor(bs)
                        actor_loss = -tf.reduce_mean(self.critic([bs, actions]))
                    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                    # Update target networks
                    self._update_target(self.target_actor, self.actor, tau=self.tau)
                    self._update_target(self.target_critic, self.critic, tau=self.tau)

                def store_transition(self, s, a, r, s_):
                    transition = np.hstack((s, a, [r], s_))
                    index = self.pointer % self.memory_capacity
                    self.memory[index, :] = transition
                    self.pointer += 1

                @property
                def memory_full(self):
                    return self.pointer >= self.memory_capacity

                def save(self, path):
                    # 確保目標路徑的資料夾存在
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    # 保存 Actor 和 Critic 模型的權重
                    self.actor.save_weights(path + '_actor.h5')
                    self.critic.save_weights(path + '_critic.h5')

                def restore(self, path):
                    self.actor.load_weights(path + '_actor.h5')
                    self.critic.load_weights(path + '_critic.h5')

            """
            with st.expander("點擊展開完整程式碼"):
                st.code(code, language="python")


            st.markdown('### 步驟3：創建main.py，匯入步驟1、2的檔案，讓手臂在設定好的環境中訓練抓取方塊')
            code = """
            ON_TRAIN = False  # 訓練時設為 True，評估時設為 False，在main.py第21行
            """
            st.code(code, language="python")        
            code = """
            from env import ArmEnv
            from rl import DDPG
            import numpy as np
            import tensorflow as tf

            # 檢查 GPU 是否可用
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

            # 啟用 GPU 記憶體增長
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth enabled")
                except RuntimeError as e:
                    print(e)

            MAX_EPISODES = 500
            MAX_EP_STEPS = 200
            ON_TRAIN = False  # 訓練時設為 True，評估時設為 False

            # Set environment
            env = ArmEnv()
            state_dim = env.state_dim
            action_dim = env.action_dim
            action_bound = env.action_bound

            # Set RL method
            rl = DDPG(state_dim, action_dim, action_bound=[1, 1, 1])  # 修改為三維

            def train():
                for episode in range(MAX_EPISODES):
                    s = env.reset()
                    ep_reward = 0
                    for step in range(MAX_EP_STEPS):
                        env.render()  # 確保每步都渲染視窗
                        a = rl.choose_action(s)
                        s_, r, done = env.step(a)
                        rl.store_transition(s, a, r, s_)

                        if rl.memory_full:
                            rl.learn()

                        s = s_
                        ep_reward += r
                        if done:
                            break

                    print(f'Episode: {episode}, Reward: {ep_reward:.2f}')

                rl.save('model/ddpg.ckpt')

            def eval():
                rl.actor.load_weights('model/ddpg.ckpt_actor.h5')
                rl.critic.load_weights('model/ddpg.ckpt_critic.h5')
                while True:
                    s = env.reset()
                    for _ in range(MAX_EP_STEPS):
                        env.render()
                        a = rl.choose_action(s)
                        s, _, done = env.step(a)
                        if done:
                            break

            if ON_TRAIN:
                train()
            else:
                eval()

            """
            with st.expander("點擊展開完整程式碼"):
                st.code(code, language="python")
            st.video("Pictures + Videos/threelink/train_threelink.mp4")    

    elif submenu == "2.5 streamlit UI設計與資料可視化":
            st.title("2.5 streamlit UI設計與資料可視化")
            st.markdown('### 步驟1：安裝streamlit套件')
            code = """
            pip install streamlit
            """
            st.code(code, language="python")

            st.markdown('### 步驟2：啟動streamlit，開啟網頁') 
            code = """
            streamlit run "01.py"  #01.py為範例檔，可檔名更改更改
            """
            st.code(code, language="python")   

            st.markdown('### 步驟3：開啟"[streamlit](https://cheat-sheet.streamlit.app/)"官網的程式庫，從中可獲的各種程式以供網頁書寫') 


elif menu == "第三章 結果與討論":
    submenu = st.sidebar.radio("關於選單", ["3.1Turtlebot3(Burger)避障與導航測試結果", "3.2 ROS/AMR之深度學習影像移動控制結果", "3.3 ultralytics YOLO與SAM物件偵測與分割結果", "3.4 three-link planar manipulator模擬結果", "3.5 streamlit UI設計與資料可視化結果"])
    if submenu == "3.1Turtlebot3(Burger)避障與導航測試結果":
        st.title('3.1Turtlebot3(Burger)避障與導航測試結果')
        st.markdown('### 成功導航使Turtlebot3到目的地，並且避開障礙物')
        st.video("Pictures + Videos/ROS_navigation/final.mp4")

    elif submenu == "3.2 ROS/AMR之深度學習影像移動控制結果":
        st.title("3.2 ROS/AMR之深度學習影像移動控制結果")
        st.markdown('### 成功使Turtlebot3依箭頭方向做出相應動作')
        st.markdown('### (a)左轉')
        st.video("Pictures + Videos/YOLO+ROS_Turtlebot/L.mp4")
        st.markdown('### (b)右轉')
        st.video("Pictures + Videos/YOLO+ROS_Turtlebot/R.mp4")
        st.markdown('### (c)前進')
        st.video("Pictures + Videos/YOLO+ROS_Turtlebot/S.mp4")

    elif submenu == "3.3 ultralytics YOLO與SAM物件偵測與分割結果":
        st.title("3.3 ultralytics YOLO與SAM物件偵測與分割結果")
        st.markdown('### 成功利用ultralytics YOLO物件辨識功能判別杏鮑菇、芭樂與香蕉，並進行SAM分割')
        st.markdown('#### 辨識杏鮑菇、並進行SAM分割')
        st.image("Pictures + Videos/YOLO+SAM/SMA_K (1).png")
        st.markdown('#### 辨識芭樂、並進行SAM分割')
        st.image("Pictures + Videos/YOLO+SAM/SAM_G.png")
        st.markdown('#### 辨識香蕉、並進行SAM分割')
        st.image("Pictures + Videos/YOLO+SAM/SAM_B (2).png")
        st.image("Pictures + Videos/YOLO+SAM/SAM_B (1).png")

    elif submenu == "3.4 three-link planar manipulator模擬結果":
        st.title("3.4 three-link planar manipulator模擬結果")
        st.markdown('### (a)ROS moveit')
        st.video("Pictures + Videos/threelink/rosmove.mp4")  
        st.markdown('### (b)RL強化學習訓練，手臂可抓取方塊')
        st.video("Pictures + Videos/threelink/final_threeink.mp4")  

    elif submenu == "3.5 streamlit UI設計與資料可視化結果":
        st.title("3.5 streamlit UI設計與資料可視化結果")
        st.markdown('### 點擊可直接開啟"[期末報告網頁](https://cheat-sheet.streamlit.app/)"查看')  
        st.markdown('### 也可透過streamlit啟動')  
        code = """
            streamlit run "Team1_Final report.py"
            """
        st.code(code, language="python")
        st.video("Pictures + Videos/streamlit UI/s1.mp4")


elif menu == "第四章 結論與心得":
    submenu = st.sidebar.radio("關於選單", ["結論", "心得"])
    if submenu == "結論":
        st.title('結論')
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本次期末報告「項目一：利用burger或waffler平台，完成自主導航與避障」、
                「項目三：利用ultralytics YOLO物件辨識功能判別三種蔬果(杏鮑菇、及其他兩種自選類別)，並能進行SAM分割」、
                「項目四：完成three-link planar manipulator之模擬：(a)ROSmoveit，(b)RL強化學習訓練」與
                「項目五：Streamlit UI設計：將功能透過網頁進行整合操作與測試」
                等四項題目成功完成實作。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                唯獨「項目二：深度學習分類問題：利用rviz+webcam，可以偵測出(a)左轉，(b)右轉，(c)前進，(d)後退及(e)STOP等動作」裡的
                (d)後退及(e)STOP等兩個動作沒有完成，其中(d)後退的功能是有辨識成功後退符號但Turtlebot3沒有做出對應的倒車動作，經檢查程式後
                確認前進與後退的相關程式差別只在輸出給馬達的訊號是否為正負值(正值為馬達正轉、反之反轉)，故研判若要馬達反轉應不單只是給它一個
                負值而已，Turtlebot3有內建保護程式，如防撞、馬達轉數上限等等的保護措施，應檢查是否有相關保護程式與反轉馬達的指令衝突，導致
                反轉馬達指令被覆蓋，如此應可解決問題，但本報告最後還是沒有成功解決，需再花時間研究。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                而(e)STOP則是辨識STOP符號失敗，但由於測試時使用webcam偵測這五個動作都可以達到九成以上的辨識成功率，因此可以排除自訓練的辨識
                模型是否辨識成功率不夠的問題，且Turtlebot3上外掛的鏡頭偵測在紅色物體佔據鏡頭畫面時會觸發STOP指令，故研判偵測失敗可能是因為
                Turtlebot3上外掛的鏡頭型號較為老舊，加上訓練模型時，STOP的照片多為紅白配色，有強烈的對比，因此Turtlebot3上外掛的鏡頭偵測到
                黑白的STOP符號時，因畫質不高加上顏色落差過大導致辨識失敗。
                </p>
                """, unsafe_allow_html=True)

    elif submenu == "心得":
        st.title("11373106/陳彥碩/心得")
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                上完這學期的智慧機械設計課程後，我覺得我收穫良多，學習到了許多大學時期沒有學習到的新知識，其中最顯著的差別就是我的
                程式能力跟以前比起來算是大幅提升了，雖然依然還有很多的不足，但現在至少可以看的懂程式碼並撰寫，也可以和別人討論交換想法，
                這是以前的我所辦不到的事情，還記得以前大學時期遇到程式相關的問題都只能依賴同學的幫忙才能完成作業解決問題，但現在我已經
                可以獨自作業並完成題目，相信今年暑假時的我是一定要辦不到的，經過這一學期的努力，我在期末報告的每個項目的程式問題都可以幫
                得上忙，雖然還很有進步的空間，但相信只要繼續努力，我也可以更有工程師的樣子。
                </p>
                """, unsafe_allow_html=True)

        st.title("11373122/林群佑/心得")
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                技術實現與效能挑戰：這次成功整合了 YOLO 和 SAM，實現了即時目標偵測和分割，但在執行過程中，因為 SAM 模型需要大量計算資源，
                我的電腦運行時直接卡住了。這也提醒我，使用深度學習模型時，硬體性能的限制是一個不可忽視的問題。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                模型選擇的取捨：為了確保偵測速度，我選擇了輕量化的 YOLOv8n，但 SAM 模型（尤其是用 vit_h 版本）對硬體需求相當高。
                未來可能需要改用更輕量的分割模型（例如 vit_b 版本），或者對模型進行壓縮，來減少對運算資源的依賴。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                過程中的收穫：儘管遇到卡頓問題，我仍然學到了如何進行深度學習模型的整合，以及影像處理管線中的關鍵步驟，包括：模型推論
                （YOLO 負責偵測，SAM 負責分割）遮罩與框的尺寸匹配及時檢測與分割畫面的整體流程。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                性能優化的必要性：這次的卡頓經驗讓我更加意識到，開發應用時性能優化真的很重要。無論是升級硬體資源，還是採用更高效的演算法
                ，都能對實際效果帶來明顯的提升。
                </p>
                """, unsafe_allow_html=True)

        st.title("11373137/廖翊翔/心得")
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                在這次實作中，我們結合了Ultralytics YOLO與Segment Anything Model（SAM），探索了物件偵測與影像分割的應用。YOLO以其快速
                、高效的目標偵測能力為基礎，準確識別影像中的物體位置與類別，而SAM則補充了精細的像素級分割能力，讓結果更具細緻性。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                透過YOLO，我們能快速完成模型訓練，並取得精確的邊界框偵測結果。接著，將這些結果與SAM整合，SAM能自動生成分割遮罩，實現高效
                的多物體分割。這種流程不僅提高了處理效率，也保證了偵測與分割的準確性。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                整體來說，YOLO的速度與SAM的細緻度相輔相成，讓物件偵測與分割流程更加高效與智能化。這次實作不僅讓我們更熟悉這兩種工具，
                也體會到AI技術整合帶來的便利與潛力。未來，我希望將這些技術應用到更多場景，如自動駕駛與醫學影像分析。
                </p>
                """, unsafe_allow_html=True)


elif menu == "參考文獻":
    st.title("參考文獻")
    st.markdown('### 1.[streamlit程式庫](https://cheat-sheet.streamlit.app/)')
    st.markdown('### 2.[USB_cam安裝](https://github.com/as985643/usb_cam.git)')
    st.markdown('### 3.[Turtlebot3影像辨識](https://emanual.robotis.com/docs/en/platform/turtlebot3/autonomous_driving/#getting-started)')
    st.markdown('### 4.[CUDA安裝方法](https://qqmanlin.medium.com/cuda-%E8%88%87-cudnn-%E5%AE%89%E8%A3%9D-e982d92162af)')
    st.markdown('### 5.[PyTorch安裝](https://pytorch.org/get-started/previous-versions/)')
    st.markdown('### 6.[Segment Anything in High Quality (HQ-SAM)](https://www.youtube.com/watch?v=UGlEU52wGwM&list=PLGTvxhgE_-gZKRbYQlE6HyUQQZ0wnYMm4&index=1)')
    st.markdown('### 7.[Segment Anything Model 2](https://www.youtube.com/watch?v=toFiUqjWCFw&list=PLGTvxhgE_-gZKRbYQlE6HyUQQZ0wnYMm4&index=3)')
    st.markdown('### 8.[Install a USB Camera in TurtleBot3](https://www.youtube.com/watch?v=hH6ov9Ep134&list=PLGTvxhgE_-gZKRbYQlE6HyUQQZ0wnYMm4&index=3)')


elif menu == "分工表":
    st.markdown("""
        <style>
            .center {
                text-align: center;
            }
            
            table {
                margin: 0 auto;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: center;
            }
            .signature {
                text-align: right;
            }
            br {
                font-size: 20px;
                }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="center">
            <h3>『智慧機械設計』</h2>
            <h4>學期團隊作業/專案設計</h3>
        </div>

        <p class="center">
            課號：0237(碩設計一甲) <br>
            學年：113年度第1學期<br>
            組別：第一組<br>
            題目：ROS自主移動平台與AI整合之研究<br>
            成員：11373106/陳彥碩、11373122/林群祐、11373137/廖翊翔<br>
        </p>

        <div class="center">
            <table>
                <tr>
                    <th>項次</th>
                    <th>學號</th>
                    <th>姓名</th>
                    <th>分工內容</th>
                    <th>貢獻度</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>11373106</td>
                    <td>陳彥碩</td>
                    <td>全篇主要程式架構撰寫、除錯與網頁報告統整</td>
                    <td>40%</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td>11373122</td>
                    <td>林群祐</td>
                    <td>YOLO的照片標註、報告討論、成果攝影與拍照</td>
                    <td>30%</td>
                </tr>
                <tr>
                    <td>3</td>
                    <td>11373137</td>
                    <td>廖翊翔</td>
                    <td>YOLO的照片標註、程式討論、步驟截圖與報告構思</td>
                    <td>30%</td>
                </tr>
            </table>
        </div>

        <p class="center">
            貢獻度總計為100%，請自行核算。<br>
            完成日期：<u>113年01月08日</u>
        </p>

        <p class="center">
            <b>說明</b><br>
            本人在此聲明，本設計作業皆由本人與同組成員共同獨立完成，並無其他第三者參與作業之進行，
            若有抄襲或其他違反正常教學之行為，自願接受該次成績以零分計。同時本人亦同意在上述表格中所記載之作業貢獻度，
            並以此計算本次個人作業成績。
        </p>

        <p class="indent">
            成員簽名：
        </p>
        """, unsafe_allow_html=True)
    st.image("Pictures + Videos/streamlit UI/1.png")
    st.image("Pictures + Videos/streamlit UI/2.png")
    st.image("Pictures + Videos/streamlit UI/3.png")
    

