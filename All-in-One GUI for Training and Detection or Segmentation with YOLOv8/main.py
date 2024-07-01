# version 0.1.1
import os
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, IntVar, Label, messagebox
from PIL import Image, ImageTk
import ctypes
import threading
import subprocess
import datetime
from queue import Queue, Empty
from src.train import create_yaml

project_name = ""
train_data_path = ""
model_save_path = ""
selected_model_size = ""
input_size = ""
epochs = ""
class_names = []
image_paths = []
current_image_index = 0
image_label = None

global start_train_button, detection_progress_bar, image_index_label, camera_detection, detection_model_path, detection_save_dir, camera_id_entry

def get_screen_size():
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height

def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def read_output(process, queue):
    for line in iter(process.stdout.readline, b''):
        queue.put(line.decode('utf-8'))
    process.stdout.close()

def on_sidebar_select(window_title):
    clear_frame(main_frame)
    if window_title == "Object_detection_traning":
        show_ai_train_window_det()
    elif window_title =="Instance_segmention_traning":
        show_ai_train_window_seg()
        

output_queue = Queue()

def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

def update_output_textbox():
    try:
        line = output_queue.get_nowait()
        output_textbox.insert("end", line)
        output_textbox.yview_moveto(1)
    except Empty:
        pass
    finally:
        root.after(100, update_output_textbox)

def update_image():
    global current_image_index, image_label, image_paths, image_index_label
    if image_paths:
        image_index_text = f"{current_image_index + 1}/{len(image_paths)}"
        image_index_label.configure(text=image_index_text)
        img = Image.open(image_paths[current_image_index])
        img_w, img_h = img.size
        max_w, max_h = image_label.winfo_width(), image_label.winfo_height()  # Use the size of the image_label widget
        scale_w = max_w / img_w
        scale_h = max_h / img_h
        scale = min(scale_w, scale_h)
        img = img.resize((int(img_w * scale), int(img_h * scale)), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo

def start_training_and_capture_output(yaml_path):
    global project_name, class_names, input_size, batch_size, epochs, model_save_path

    def run_training():
        nonlocal process
        if not all([project_name, train_data_path, class_names, model_save_path, selected_model_size, input_size, epochs, batch_size]):
            print("Error: One or more required parameters are missing.")
            return

        cmd_args = [
            'python', 'src/train.py',
            project_name, train_data_path, ','.join(class_names),
            model_save_path, selected_model_size, str(input_size),
            str(epochs), yaml_path, str(batch_size)
        ]

        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        threading.Thread(target=enqueue_output, args=(process.stdout, output_queue), daemon=True).start()
        process.wait()
        progress_bar.stop()

    process = None
    threading.Thread(target=run_training, daemon=True).start()
    progress_bar.start()
    
print("****************************************************object detection***************************************************************")
def show_ai_train_window_det():
    global project_name_entry, input_size_entry, epochs_entry, batch_size_entry, class_names_text, progress_bar, model_size_var, output_textbox, start_train_button
    # Place GUI elements in the AI ​​creation window
    main_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

    # Project name (half-width alphanumeric characters)
    ctk.CTkLabel(master=main_frame, text="Project name (half-width alphanumeric characters)", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.03, anchor=ctk.CENTER)
    project_name_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Project Name", width=250, height=50, font=("Roboto Medium", 18))
    project_name_entry.place(relx=0.16, rely=0.06, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # Training data selection button
    ctk.CTkLabel(master=main_frame, text="Select training data", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.10, anchor=ctk.CENTER)
    train_data_button = ctk.CTkButton(master=main_frame, text="Select Train Data", command=select_train_data, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    train_data_button.place(relx=0.16, rely=0.13, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # Model save destination selection button
    ctk.CTkLabel(master=main_frame, text="Selecting where to save the model", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.17, anchor=ctk.CENTER)
    model_save_button = ctk.CTkButton(master=main_frame, text="Select Model's Save Folder", command=select_model_save_folder, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    model_save_button.place(relx=0.16, rely=0.2, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)


    # YOLOv8 model size
    ctk.CTkLabel(master=main_frame, text="YOLOv8 model size selection", font=("Roboto Medium", 18)).place(relx=0.09, rely=0.32, anchor=tk.W)
    initial_relx_v8 = 0.02
    step_size_v8 = 0.05
    model_sizes_v8 = [("Nano", "n", 1), ("Small", "s", 2), ("Medium", "m", 3), ("Large", "l", 4), ("ExtraLarge", "x", 5)]
    for index, (text, model_code, value) in enumerate(model_sizes_v8, start=1):
        ctk.CTkRadioButton(master=main_frame, text=text, variable=model_size_var, value=value, fg_color='deep sky blue').place(relx=initial_relx_v8 + step_size_v8 * (index - 1), rely=0.34)

    # Specifying the size of CNN input layer
    ctk.CTkLabel(master=main_frame, text="CNN input layer size [Example: 640]", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.39, anchor=ctk.CENTER)
    input_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Input Size", font=("Roboto Medium", 18))
    input_size_entry.place(relx=0.16, rely=0.42, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # number of epochs
    ctk.CTkLabel(master=main_frame, text="Number of epochs [Example: 100]", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.46, anchor=ctk.CENTER)
    epochs_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Epochs", font=("Roboto Medium", 18))
    epochs_entry.place(relx=0.16, rely=0.49, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # batch size
    ctk.CTkLabel(master=main_frame, text="Batch size [Example: 16]", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.53, anchor=ctk.CENTER)
    batch_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Batch size", font=("Roboto Medium", 18))
    batch_size_entry.place(relx=0.16, rely=0.56, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # Class name input window
    ctk.CTkLabel(master=main_frame, text="Enter class name", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.60, anchor=ctk.CENTER)
    class_names_text = ctk.CTkTextbox(master=main_frame, font=("Roboto Medium", 18))
    class_names_text.place(relx=0.16, rely=0.7, relwidth=0.3, relheight=0.17,  anchor=ctk.CENTER)

    # Start learning button
    start_train_button = ctk.CTkButton(master=main_frame, text="Start_Detection_Training!", command=start_training_det, fg_color="chocolate1",border_color='black', border_width=3, font=("Roboto Medium", 25, "bold"), text_color='white')
    start_train_button.place(relx=0.16, rely=0.84, relwidth=0.3, relheight=0.05, anchor=ctk.CENTER)

    # Training progress display window
    output_textbox = ctk.CTkTextbox(master=main_frame, corner_radius=20, font=("Roboto Medium", 14))
    output_textbox.place(relx=0.66, rely=0.45, relwidth=0.68, relheight=0.86, anchor=ctk.CENTER)

    # progress bar
    progress_bar = ctk.CTkProgressBar(master=main_frame, progress_color='limegreen', mode='indeterminate', indeterminate_speed=0.7)
    progress_bar.place(relx=0.5, rely=0.94, relwidth=0.7, anchor=ctk.CENTER)
    
print("**************************************************** object Segmention  ***************************************************************")

def show_ai_train_window_seg():
    global project_name_entry, input_size_entry, epochs_entry, batch_size_entry, class_names_text, progress_bar, model_size_var, output_textbox, start_train_button
    # Place GUI elements in the AI ​​creation window
    main_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

    # Project name (half-width alphanumeric characters)
    ctk.CTkLabel(master=main_frame, text="Project name (half-width alphanumeric characters)", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.03, anchor=ctk.CENTER)
    project_name_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Project Name", width=250, height=50, font=("Roboto Medium", 18))
    project_name_entry.place(relx=0.16, rely=0.06, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # Training data selection button
    ctk.CTkLabel(master=main_frame, text="Select training data", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.10, anchor=ctk.CENTER)
    train_data_button = ctk.CTkButton(master=main_frame, text="Select Train Data", command=select_train_data, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    train_data_button.place(relx=0.16, rely=0.13, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # Model save destination selection button
    ctk.CTkLabel(master=main_frame, text="Selecting where to save the model", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.17, anchor=ctk.CENTER)
    model_save_button = ctk.CTkButton(master=main_frame, text="Select Model's Save Folder", command=select_model_save_folder, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    model_save_button.place(relx=0.16, rely=0.2, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)


    # YOLOv8 model size
    ctk.CTkLabel(master=main_frame, text="YOLOv8 model size selection", font=("Roboto Medium", 18)).place(relx=0.09, rely=0.32, anchor=tk.W)
    initial_relx_v8 = 0.02
    step_size_v8 = 0.05
    model_sizes_v8 = [("Nano", "n", 1), ("Small", "s", 2), ("Medium", "m", 3), ("Large", "l", 4), ("ExtraLarge", "x", 5)]
    for index, (text, model_code, value) in enumerate(model_sizes_v8, start=1):
        ctk.CTkRadioButton(master=main_frame, text=text, variable=model_size_var, value=value, fg_color='deep sky blue').place(relx=initial_relx_v8 + step_size_v8 * (index - 1), rely=0.34)

    # Specifying the size of CNN input layer
    ctk.CTkLabel(master=main_frame, text="CNN input layer size [Example: 640]", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.39, anchor=ctk.CENTER)
    input_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Input Size", font=("Roboto Medium", 18))
    input_size_entry.place(relx=0.16, rely=0.42, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # number of epochs
    ctk.CTkLabel(master=main_frame, text="Number of epochs [Example: 100]", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.46, anchor=ctk.CENTER)
    epochs_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Epochs", font=("Roboto Medium", 18))
    epochs_entry.place(relx=0.16, rely=0.49, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # batch size
    ctk.CTkLabel(master=main_frame, text="Batch size [Example: 16]", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.53, anchor=ctk.CENTER)
    batch_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Batch size", font=("Roboto Medium", 18))
    batch_size_entry.place(relx=0.16, rely=0.56, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # Class name input window
    ctk.CTkLabel(master=main_frame, text="Enter class name", font=("Roboto Medium", 18)).place(relx=0.16, rely=0.60, anchor=ctk.CENTER)
    class_names_text = ctk.CTkTextbox(master=main_frame, font=("Roboto Medium", 18))
    class_names_text.place(relx=0.16, rely=0.7, relwidth=0.3, relheight=0.17,  anchor=ctk.CENTER)

    # Start learning button
    start_train_button = ctk.CTkButton(master=main_frame, text="Start_Segmentation_Training!", command=start_training_det, fg_color="chocolate1",border_color='black', border_width=3, font=("Roboto Medium", 25, "bold"), text_color='white')
    start_train_button.place(relx=0.16, rely=0.84, relwidth=0.3, relheight=0.05, anchor=ctk.CENTER)

    # Training progress display window
    output_textbox = ctk.CTkTextbox(master=main_frame, corner_radius=20, font=("Roboto Medium", 14))
    output_textbox.place(relx=0.66, rely=0.45, relwidth=0.68, relheight=0.86, anchor=ctk.CENTER)

    # progress bar
    progress_bar = ctk.CTkProgressBar(master=main_frame, progress_color='limegreen', mode='indeterminate', indeterminate_speed=0.7)
    progress_bar.place(relx=0.5, rely=0.94, relwidth=0.7, anchor=ctk.CENTER)


def select_train_data():
    global train_data_path
    train_data_path = filedialog.askdirectory()

def select_model_save_folder():
    global model_save_path
    model_save_path = filedialog.askdirectory()

def select_detection_images_folder():
    global detection_images_folder_path
    detection_images_folder_path = filedialog.askdirectory()
    if detection_images_folder_path:
        print(f"Selected folder: {detection_images_folder_path}")

def select_detection_model():
    global detection_model_path
    detection_model_path = filedialog.askopenfilename(filetypes=[("YOLOv8 Model", "*.pt")])
    if detection_model_path:
        print(f"Selected model: {detection_model_path}")

def select_detection_yaml():
    global detection_yaml_path
    detection_yaml_path = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
    if detection_yaml_path:
        print(f"Selected YAML: {detection_yaml_path}")

def animate_progress_bar(progress, step):
    if progress >= 100 or progress <= 0:
        step = -step

    progress_bar.set(progress)
    root.after(50, animate_progress_bar, progress + step, step)

def start_training_seg():
    global project_name, train_data_path, model_save_path, selected_model_size, input_size, epochs, batch_size, class_names
    project_name = project_name_entry.get()
    input_size = input_size_entry.get()
    epochs = epochs_entry.get()
    batch_size = batch_size_entry.get()
    class_names = class_names_text.get("1.0", "end-1c").split('\n')
    class_names = [name for name in class_names if name.strip() != '']

    model_size_options = {1:"yolov8n-seg", 2: "yolov8s-seg", 3: "yolov8m-seg", 4: "yolov8l-seg", 5: "yolov8x-seg"}
    selected_model_size = model_size_options[model_size_var.get()]

    if not all([project_name, train_data_path, model_save_path, selected_model_size, input_size, epochs, batch_size, class_names]):
        print("Error: One or more required parameters are missing.")
        return

    yaml_path = create_yaml(project_name, train_data_path, class_names, model_save_path)
    start_training_and_capture_output(yaml_path)

def start_training_det():
    global project_name, train_data_path, model_save_path, selected_model_size, input_size, epochs, batch_size, class_names
    project_name = project_name_entry.get()
    input_size = input_size_entry.get()
    epochs = epochs_entry.get()
    batch_size = batch_size_entry.get()
    class_names = class_names_text.get("1.0", "end-1c").split('\n')
    class_names = [name for name in class_names if name.strip() != '']

    model_size_options = {1:"yolov8n", 2: "yolov8s", 3: "yolov8m", 4: "yolov8l", 5: "yolov8x"}
    selected_model_size = model_size_options[model_size_var.get()]

    if not all([project_name, train_data_path, model_save_path, selected_model_size, input_size, epochs, batch_size, class_names]):
        print("Error: One or more required parameters are missing.")
        return

    yaml_path = create_yaml(project_name, train_data_path, class_names, model_save_path)
    start_training_and_capture_output(yaml_path)



def update_image_list(results_dir):
    global image_paths, current_image_index, detection_progress_bar
    image_paths = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.jpg') or f.endswith('.png')or f.endswith('.bmp')]
    current_image_index = 0
    update_image()
    detection_progress_bar.stop()

def change_appearance_mode(new_appearance_mode):
    ctk.set_appearance_mode(new_appearance_mode)
    
# Object_detection_Tool
def object_detection_btn():
    try:
        subprocess.Popen(['labelimg'])
    except Exception as e:
        print("Error:", e)
        
# Instance Segmentation_Tool
def object_segmention_btn():
    try:
        subprocess.Popen(['labelme'])
    except Exception as e:
        print("Error:", e)

screen_width, screen_height = get_screen_size()
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title('YOLO Train and Detect App')
root.geometry(f"{screen_width}x{screen_height}")
model_size_var = IntVar(value=1)

sidebar = ctk.CTkFrame(master=root, width=380, corner_radius=0)
sidebar.pack(side="left", fill="y")

main_frame = ctk.CTkFrame(master=root)
main_frame.pack(fill="both", expand=True, padx=10, pady=15)

app_name_label = ctk.CTkLabel(master=sidebar, text="YOLOv8_Detection", font=("Roboto Medium", 16))
app_name_label.pack(pady=10)

image_tool_button = ctk.CTkButton(master=sidebar, text="Detection_Tool", command=object_detection_btn, fg_color="dodgerblue", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 24))
image_tool_button.pack(pady=8)

ai_creation_button = ctk.CTkButton(master=sidebar, text="Detection_traning", command=lambda: on_sidebar_select("Object_detection_traning"), fg_color="chocolate1", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 24))
ai_creation_button.pack(pady=8)

app_name_label = ctk.CTkLabel(master=sidebar, text="YOLOv8_Segementions", font=("Roboto Medium", 16))
app_name_label.pack(pady=10)

image_tool_button = ctk.CTkButton(master=sidebar, text="Segmentation_Tool", command=object_segmention_btn, fg_color="dodgerblue", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 24))
image_tool_button.pack(pady=8)

ai_creation_button = ctk.CTkButton(master=sidebar, text="Segmentation_traning", command=lambda: on_sidebar_select("Instance_segmention_traning"), fg_color="chocolate1", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 24))
ai_creation_button.pack(pady=8)


empty_space = ctk.CTkLabel(master=sidebar, text="")
empty_space.pack(fill=tk.BOTH, expand=True)

appearance_mode_var = ctk.StringVar(value="Light")
appearance_mode_label = ctk.CTkLabel(master=sidebar, text="Appearance Mode", font=("Roboto Medium", 12))
appearance_mode_label.pack(padx=10, pady=(0, 5), anchor='w')

light_mode_radio = ctk.CTkRadioButton(master=sidebar, text="Light", variable=appearance_mode_var, value="Light", command=lambda: change_appearance_mode("Light"))
light_mode_radio.pack(padx=10, pady=(0, 5), anchor='w')

dark_mode_radio = ctk.CTkRadioButton(master=sidebar, text="Dark", variable=appearance_mode_var, value="Dark", command=lambda: change_appearance_mode("Dark"))
dark_mode_radio.pack(padx=10, pady=(0, 10), anchor='w')

signature_label = ctk.CTkLabel(master=sidebar, text="© Design by Rahul Kumar", text_color="white", font=("Roboto Medium", 12),fg_color="dodgerblue",corner_radius=30)
signature_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5, anchor='w')

if __name__ == "__main__":
    root.after(100, update_output_textbox)
    root.mainloop()
