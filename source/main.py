# This Python file uses the following encoding: utf-8
import sys
import os
import ffmpeg
import transcriber
import subprocess
import threading
import time
import shutil
from PyQt5.QtWidgets import QMainWindow, QApplication, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import  QSpacerItem, QStatusBar, QProgressBar, QPushButton, QVBoxLayout, QWidget, QAction, QMessageBox, QFileDialog, QLabel, QSizePolicy, QGroupBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QHBoxLayout
from moviepy.editor import VideoFileClip
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer
from pydub import AudioSegment
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QInputDialog, QDialogButtonBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QDialog, QHeaderView
from googletrans import Translator

# Print the PATH of ffmpeg
#ffmpeg_path = which("ffmpeg")
#print(f"FFmpeg path: {ffmpeg_path}")

#Print the PATH of Montreal Forced Aligner
#mfa_path = which("mfa")
#print(f"MFA path: {mfa_path}")

# Print the PATH environment variable
#print("Current PATH environment variable:")
#for path in os.environ['PATH'].split(os.pathsep):
#    print(path)
# def is_admin():
#     try:
#         return os.getuid() == 0
#     except AttributeError:
#         # Windows
#         return ctypes.windll.shell32.IsUserAnAdmin()
    
# def run_as_admin():
#     if not is_admin():
#         # Re-launch the script with admin rights
#         print("Requesting administrative privileges...")
#         ctypes.windll.shell32.ShellExecuteW(
#             None, "runas", sys.executable, ' '.join(sys.argv), None, 1)
        

class VideoToAudioApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def load_video(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.flv)", options=options)
        if fileName:
            #output_audio = f'{fileName.rsplit(".", 1)[0]}.wav'
            #self.extract_audio(fileName, output_audio)
            video_info = self.get_video_info(fileName)
            self.video_info_label.setText(f"Video name: {video_info['name']}\nDuration: {video_info['duration']} seconds")

    def extract_audio(self,input_video, output_audio):
        if not os.path.isfile(input_video):
            print(f'Input video file not found: {input_video}')
            return
        try:
            ffmpeg.input(input_video).output(output_audio).run()
            print(f'Audio extracted successfully: {output_audio}')
        except ffmpeg.Error as e:
            print(f'Error occurred: {e.stderr.decode()}')

    def get_video_info(self, video_path):
        video = VideoFileClip(video_path)
        return {
            'name': os.path.basename(video_path),
            'duration': video.duration
        }
    
    def initUI(self):
        self.setWindowTitle('Echo - Video to Audio Extractor')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        self.label = QLabel('Select a video file to extract audio')
        layout.addWidget(self.label)
          
        self.load_button= QPushButton('Load video')
        self.load_button.clicked.connect(self.load_video)
        layout.addWidget(self.load_button)

        #Add a progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget

        #Add a label for displaying the progess percentage
        self.progress_label=QLabel(self)
        layout.addWidget(self.progress_label)

        #Add a label for displaying the video information
        self.progress_label=QLabel(self)
        layout.addWidget(self.progress_label)

        self.video_info_label = QLabel()
        self.preview_label = QLabel()        
        layout.addWidget(self.video_info_label)
        layout.addWidget(self.preview_label)

        self.setLayout(layout)

class MainWindow(QMainWindow):
    task_completed = pyqtSignal()
    start_progress_signal = pyqtSignal(float)
    stop_timer_signal = pyqtSignal()
    open_srt_after_completion = False

    def showAboutDialog(self):
        QMessageBox.about(self, "About Echo",
        """
        Echo v1.0.0
        Author: Kyle Dang
        Copyright © 2024. All rights reserved.
        """
        )
    def show_completed_dialog(self):
        QMessageBox.information(self, "Completed", "The video has been exported successfully.", QMessageBox.Ok)

    def __init__(self):
        super().__init__()       
        
        # Set window properties
        self.setWindowTitle("Echo")
        self.setGeometry(100, 100, 800, 600)               

        #Set the translator
        self.translator = Translator()

        # Initialize UI components
        self.initUI()

        # Set window flags to disable the maximize button
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Connect the task completed signal to the progress bar update
        self.task_completed.connect(self.complete_progress_bar)
       
        # Connect to the completed dialog
        self.task_completed.connect(self.show_completed_dialog)

        #Initialize file path to an empty string
        self.video_file_path = ""
        self.srt_file_path = ""
        self.wav_file_path = ""
        self.video_file_paths = []

        #Create a status bar
        self.statusBar = QStatusBar()

        #Create a progress bar
        self.progress_bar = QProgressBar()
      
        #Disable the text display on the progess bar
        self.progress_bar.setTextVisible(False)
        
        #Set the size policy to make the progress bar expand horizontally
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        #Set a minimum height to make the progress bar bigger
        self.progress_bar.setMinimumHeight(50)

        #Create a horizontal layout
        layout = QHBoxLayout()

        # Add a spacer to the left of the progress bar
        layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Add the progress bar to the layout
        layout.addWidget(self.progress_bar)

        # Get the current working directory
        current_directory = os.getcwd()

        # Construct the full path to echo-dot.png
        image_path = os.path.join(current_directory, 'echo-dot.png')

        # Set the window icon
        self.setWindowIcon(QIcon(image_path))
        
        # Add a spacer to the right of the progress bar
        layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Create a widget to hold the layout
        layout_widget = QWidget()
        layout_widget.setLayout(layout)

        # Add the widget to the status bar
        self.statusBar.addWidget(layout_widget, 1)

        # Set the status bar of the window
        self.setStatusBar(self.statusBar)

        # Move the status bar up a little bit
        self.statusBar.setContentsMargins(0, -40, 0, 10)
        
        #Filename inizialization
        self.fileName=""
    
        
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit',
                                     'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if os.path.exists("run"):
                shutil.rmtree("run")
            event.accept()
        else:
            event.ignore()    
    """
    def load_video(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.flv)", options=options)

        # Check if files were selected
        if not files:
            print("No files selected. Keeping the application open.")
            return  # Exit the function if no files were selected

        self.video_file_paths = files  # Store the list of selected video files

        # Process the first selected video file
        fileName = files[0]
        video_to_audio_app = VideoToAudioApp()
        output_audio = f'{fileName.rsplit(".", 1)[0]}.wav'

        if os.path.exists(output_audio):
            os.remove(output_audio)

        # Get video information
        try:
            video_info = video_to_audio_app.get_video_info(fileName)
        except Exception as e:
            print(f"Error getting video info for {fileName}: {e}")
            return  # Exit the function if there is an error

        # Close the previous video clip if it exists
        if hasattr(self, 'video_clip') and self.video_clip is not None:
            self.video_clip.close()
            self.video_clip = None

        # Load the new video clip
        self.video_clip = VideoFileClip(fileName)

        # Display the preview picture of the first video
        preview_frame = self.video_clip.get_frame(1)
        preview_image = QImage(preview_frame, preview_frame.shape[1], preview_frame.shape[0], preview_frame.shape[1] * 3, QImage.Format_RGB888)

        # Create a QLabel for the preview image and set it as a child of video_box
        self.preview_label.setPixmap(QPixmap(preview_image))
        self.preview_label.setScaledContents(True)

        # Resize the QLabel to maintain the aspect ratio of the preview image
        aspect_ratio = preview_frame.shape[1] / preview_frame.shape[0]
        widget_width = self.video_widget.width()
        widget_height = int(widget_width / aspect_ratio)
        self.preview_label.setFixedSize(widget_width, widget_height)

        print("Videos loaded successfully.")
    """
    def load_video(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.flv)", options=options)

        # Check if files were selected
        if not files:
            print("No files selected. Keeping the application open.")
            return  # Exit the function if no files were selected

        self.video_file_paths = files  # Store the list of selected video files

        # Process the first selected video file
        fileName = files[0]
        video_to_audio_app = VideoToAudioApp()
        output_audio = f'{fileName.rsplit(".", 1)[0]}.wav'

        if os.path.exists(output_audio):
            os.remove(output_audio)

        # Get video information
        try:
            video_info = VideoToAudioApp.get_video_info(self,fileName)
        except Exception as e:
            print(f"Error getting video info for {fileName}: {e}")
            return  # Exit the function if there is an error

        # Close the previous video clip if it exists
        if hasattr(self, 'video_clip') and self.video_clip is not None:
            self.video_clip.close()
            self.video_clip = None

        # Load the new video clip
        self.video_clip = VideoFileClip(fileName)

        # Display the preview picture of the first video
        preview_frame = self.video_clip.get_frame(1)
        preview_image = QImage(preview_frame, preview_frame.shape[1], preview_frame.shape[0], preview_frame.shape[1] * 3, QImage.Format_RGB888)
        
        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(preview_image)

        # Set the QPixmap to the QLabel inside the frame
        self.icon_button.setIcon(QIcon(pixmap))
        self.icon_button.setIconSize(self.video_frame.size())        
        
        # Resize the QLabel to fit the frame
        self.icon_button.setFixedSize(self.video_frame.size())
        
        #Hide the frame
        self.video_frame.setStyleSheet("background: transparent; border: none;")  # Remove the frame border

        print("Video loaded successfully.")

    def start_progress_bar(self, video_duration):
        self.progress_bar.setValue(0)
        self.progress_value = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress_bar)
        self.estimated_time = video_duration *1.5 * 1000  # Convert to milliseconds
        self.timer.start(self.estimated_time / 90)  # Update every 1% until 90%

    def update_progress_bar(self):
        if self.progress_value < 90:
            self.progress_value += 1
            self.progress_bar.setValue(self.progress_value)
        #else:
        #   self.stop_timer_signal.emit()

    @pyqtSlot()
    #def stop_timer(self):
    #    self.timer.stop()
        
    def complete_progress_bar(self):        
        self.progress_bar.setValue(100)
        if hasattr(self, 'timer'):
            self.timer.stop()

    def ask_language(self):
        languages = ["English", "Vietnamese"]
        language, ok = QInputDialog.getItem(self, "Select Language", "Please choose your subtitle language!", languages, 0, False)
        if ok and language:
            return language
        else:
            return None
    
    def translate_srt(self, file_path, target_language):
        print(f"Translating SRT file to {target_language}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        translated_lines = []
        for line in lines:
            if line.strip() and not line.strip().isdigit() and '-->' not in line:
                translated_line = self.translator.translate(line, dest=target_language).text
                translated_lines.append(translated_line + '\n')
            else:
                translated_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(translated_lines)

        print(f"SRT file translated to {target_language} successfully.")  

    def translate_text(self, text, target_language):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return translated_text
    
    def ask_open_file(self):
        reply = QMessageBox.question(self, "Open File", "Open the output folder after complete?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.open_srt_after_completion = True

        return reply == QMessageBox.Yes
    
    

    def export_single_subtitle(self, language):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "Subtitle Files (*.srt)", options=options)

        if fileName:
            if not fileName.lower().endswith('.srt'):
                fileName += '.srt'

            self.start_progress_bar(self.video_clip.duration)

            start_time = time.time()                 
            def task():
                # Convert MP4 to WAV
                try:
                    audio = AudioSegment.from_file(self.video_file_paths[0], format="mp4")
                    wav_file_path = os.path.splitext(self.video_file_paths[0])[0] + '.wav'
                    audio.export(wav_file_path, format="wav")

                    transcriber.generate_srt(wav_file_path, fileName)
                    

                    if language == "Vietnamese":
                        self.translate_srt(fileName, 'vi')
                    
                    end_time = time.time()
                    elapsed_time = (end_time - start_time) / 60
                    print(f"Time taken to generate subtitle: {elapsed_time:.2f} minutes")

                    # Check if the user wants to open the SRT file after generation
                    if self.open_srt_after_completion:
                        os.startfile(os.path.dirname(fileName))
                finally:
                    self.task_completed.emit()
            threading.Thread(target=task).start()

    def get_first_frame(self, video_file_path):
                # Load the video clip
                video_clip = VideoFileClip(video_file_path)
    
                # Get the first frame
                preview_frame = video_clip.get_frame(0)  # Use 0 to get the first frame
    
                # Convert the frame to QImage
                preview_image = QImage(preview_frame, preview_frame.shape[1], preview_frame.shape[0], preview_frame.shape[1] * 3, QImage.Format_RGB888)
    
                # Close the video clip to release resources
                video_clip.close()
    
                return preview_image
    
    def export_multiple_subtitles(self, language):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            QMessageBox.critical(self, "Error", "You must select an output folder.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Video Order")
        dialog.resize(800,600)
        layout = QVBoxLayout(dialog)

        table = QTableWidget(len(self.video_file_paths), 2)
        table.setHorizontalHeaderLabels(["Video Name", "Output SRT File Name"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, video_path in enumerate(self.video_file_paths):
            video_name = os.path.basename(video_path)
            srt_base_name=os.path.splitext(video_name)[0]

            # Set the video name as read-only
            video_item = QTableWidgetItem(video_name)
            video_item.setFlags(video_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 0, video_item)

            # Set the output SRT file name as editable, but display the .srt extension
            srt_item = QTableWidgetItem(srt_base_name)
            table.setItem(i, 1, srt_item)

        layout.addWidget(table)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.Accepted:
            new_order = []
            output_file_names = []  # List to store output file names
            for i in range(table.rowCount()):
                video_name = table.item(i, 0).text()
                output_file_name = table.item(i, 1).text()  # Assuming the output file name is in the second column
                output_file_names.append(output_file_name)
                for path in self.video_file_paths:
                    if os.path.basename(path) == video_name:
                        new_order.append(path)
                        break
            self.video_file_paths = new_order
            
            
            # Calculate the total duration of all selected videos
            total_duration = 0
            for video_path in self.video_file_paths:
                video_clip = VideoFileClip(video_path)
                total_duration += video_clip.duration

            # Start the progress bar with the total duration
            self.start_progress_bar(total_duration)                 


            def task():
                try:
                    total_videos = len(self.video_file_paths)
                    for i, video_path in enumerate(self.video_file_paths): # Update the preview picture based on the current video in the loop
                        preview_image = self.get_first_frame(video_path)
                        self.preview_label.setPixmap(QPixmap(preview_image))
                        self.preview_label.setScaledContents(True)

                        # Resize the QLabel to maintain the aspect ratio of the preview image
                        aspect_ratio = preview_image.width() / preview_image.height()
                        widget_width = self.video_widget.width()
                        widget_height = int(widget_width / aspect_ratio)
                        self.preview_label.setFixedSize(widget_width, widget_height)
                     

                        # Start progress bar for each video
                        output_file_name = output_file_names[i]
                        srt_path = os.path.join(folder, output_file_name + '.srt')

                        # Convert MP4 to WAV
                        audio = AudioSegment.from_file(video_path, format="mp4")
                        
                        # Get the directory where the script is currently running
                        current_working_dir = os.path.dirname(os.path.abspath(__file__))
                        run_folder_dir = os.path.join(current_working_dir, "run")
    
                        if os.path.exists(run_folder_dir):
                            shutil.rmtree(run_folder_dir)

                        os.makedirs(run_folder_dir, exist_ok=True)

                        wav_file_path = os.path.join(run_folder_dir, os.path.splitext(os.path.basename(video_path))[0] + '.wav')
                        audio.export(wav_file_path, format="wav")

                        transcriber.generate_srt(wav_file_path, srt_path)


                        if language == "Vietnamese":
                            self.translate_srt(srt_path, 'vi')       
                finally:
                    self.task_completed.emit()
                    # Check if the user wants to open the SRT file after generation
                    if self.open_file_after_completion:
                        os.startfile(folder)

            # Create and start the thread
            threading.Thread(target=task).start()
            #end_time = time.time()
            #elapsed_time = (end_time - start_time) / 60
            #print(f"Time taken to generate subtitle: {elapsed_time:.2f} minutes")
        

    def export_subtitle(self):
        language = self.ask_language()
        if not language:
            QMessageBox.critical(self, "Error", "You must select a language.")
            return

        self.open_file_after_completion = self.ask_open_file()

        if not self.video_file_paths:
            QMessageBox.critical(self, "Error", "You must open a video file before saving subtitles.")
            return

        if len(self.video_file_paths) == 1:
          self.export_single_subtitle(language)
        else:
            self.export_multiple_subtitles(language)  
        
        
             
    def export_single_video(self, language):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "Video Files (*.mp4)", options=options)

        if fileName:
            if not fileName.lower().endswith('.mp4'):
                fileName += '.mp4'

            self.start_progress_bar(self.video_clip.duration)
            srt_file_name = os.path.splitext(fileName)[0] + '.srt'            
            
            
            start_time = time.time()                 
            def task():
                # Convert MP4 to WAV
                try:
                    # Get the file extension of the input video
                    video_file_path = self.video_file_paths[0]
                    file_extension = os.path.splitext(video_file_path)[1][1:]  # Remove the dot from the extension

             
                    # Get the directory where the script is currently running
                    current_working_dir = os.path.dirname(os.path.abspath(__file__))
                    run_folder_dir = os.path.join(current_working_dir, "run")
    
                    if os.path.exists(run_folder_dir):
                        shutil.rmtree(run_folder_dir)

                    os.makedirs(run_folder_dir, exist_ok=True)
                    
                    # Check if the file extension is not MP4
                    if file_extension.lower() != 'mp4':
                        converted_video_path = os.path.join(run_folder_dir, 'converted_video.mp4')
                        cmd = [
                            'HandBrakeCLI',
                            '-i', video_file_path,
                            '-o', converted_video_path
                        ]
                        subprocess.run(cmd)
                        video_file_path = converted_video_path  # Update the video file path to the converted video

                    audio = AudioSegment.from_file(video_file_path, format="mp4")
                    wav_file_path = os.path.splitext(self.video_file_paths[0])[0] + '.wav'
                    audio.export(wav_file_path, format="wav")

                    transcriber.generate_srt(wav_file_path, srt_file_name)
                    os.remove(wav_file_path)

                    if language == "Vietnamese":
                        self.translate_srt(srt_file_name, 'vi')                    

                    cmd = [
                        'HandBrakeCLI',
                        '-i', self.video_file_paths[0],
                        '-o', fileName,
                        '--srt-file', srt_file_name,
                        '--srt-codeset', 'UTF-8',
                        '--srt-burn=1'
                    ]

                    subprocess.run(cmd)
                    print(f'Video exported successfully: {fileName}')
                    os.remove(srt_file_name) 

                    end_time = time.time()
                    elapsed_time = (end_time - start_time) / 60
                    print(f"Time taken to generate subtitle: {elapsed_time:.2f} minutes")

                    # Check if the user wants to open the SRT file after generation
                    if self.open_srt_after_completion:
                        os.startfile(os.path.dirname(fileName))
                finally:
                    self.task_completed.emit()
                    
            threading.Thread(target=task).start()
            

    def export_multiple_videos(self, language):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            QMessageBox.critical(self, "Error", "You must select an output folder.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Video Order")
        dialog.resize(800,600)
        layout = QVBoxLayout(dialog)

        table = QTableWidget(len(self.video_file_paths), 2)
        table.setHorizontalHeaderLabels(["Video Name", "Output MP4 File Name"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, video_path in enumerate(self.video_file_paths):
            video_name = os.path.basename(video_path)
            mp4_base_name = os.path.splitext(video_name)[0]

            # Set the video name as read-only
            video_item = QTableWidgetItem(video_name)
            video_item.setFlags(video_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 0, video_item)

            # Set the output MP4 file name as editable, but display the .mp4 extension
            mp4_item = QTableWidgetItem(mp4_base_name)
            table.setItem(i, 1, mp4_item)

        layout.addWidget(table)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.Accepted:
            new_order = []
            output_file_names = []  # List to store output file names

            for i in range(table.rowCount()):
                video_name = table.item(i, 0).text()
                output_file_name = table.item(i, 1).text()  # Assuming the output file name is in the second column
                output_file_names.append(output_file_name)
                for path in self.video_file_paths:
                    if os.path.basename(path) == video_name:
                        new_order.append(path)
                        break
            self.video_file_paths = new_order
            
            
            # Calculate the total duration of all selected videos
            total_duration = 0
            for video_path in self.video_file_paths:
                video_clip = VideoFileClip(video_path)
                total_duration += video_clip.duration

            # Start the progress bar with the total duration
            self.start_progress_bar(total_duration)                 


            def task():
                try:
                    total_videos = len(self.video_file_paths)
                    for i, video_path in enumerate(self.video_file_paths): # Update the preview picture based on the current video in the loop
                        """
                        preview_image = self.get_first_frame(video_path)
                        self.preview_label.setPixmap(QPixmap(preview_image))
                        self.preview_label.setScaledContents(True)

                        # Resize the QLabel to maintain the aspect ratio of the preview image
                        aspect_ratio = preview_image.width() / preview_image.height()
                        widget_width = self.video_widget.width()
                        widget_height = int(widget_width / aspect_ratio)
                        self.preview_label.setFixedSize(widget_width, widget_height)
                        """                       
                        # Load the new video clip
                        self.video_clip = VideoFileClip(self.video_file_paths[i])
                       
                        preview_frame = self.video_clip.get_frame(1)

                        preview_image = QImage(preview_frame, preview_frame.shape[1], preview_frame.shape[0], preview_frame.shape[1] * 3, QImage.Format_RGB888)
        

                        # Create a QPixmap from the QImage
                        pixmap = QPixmap.fromImage(preview_image)

                        # Set the QPixmap to the QLabel inside the frame
                        self.icon_button.setIcon(QIcon(pixmap))
                        self.icon_button.setIconSize(self.video_frame.size())        
        
                        # Resize the QLabel to fit the frame
                        self.icon_button.setFixedSize(self.video_frame.size())
        
                        #Hide the frame
                        self.video_frame.setStyleSheet("background: transparent; border: none;")  # Remove the frame border

                     
                        # Start progress bar for each video
                        output_file_name =os.path.join(folder,output_file_names[i] + '.mp4')
                        srt_path = os.path.join(folder, output_file_name + '.srt')

                        # Get the file extension of the input video
                        video_file_path = self.video_file_paths[0]
                        file_extension = os.path.splitext(video_file_path)[1][1:]  # Remove the dot from the extension

                        # Get the directory where the script is currently running
                        current_working_dir = os.path.dirname(os.path.abspath(__file__))
                        run_folder_dir = os.path.join(current_working_dir, "run")
    
                        if os.path.exists(run_folder_dir):
                            shutil.rmtree(run_folder_dir)

                        os.makedirs(run_folder_dir, exist_ok=True)
                    
                        # Check if the file extension is not MP4
                        if file_extension.lower() != 'mp4':
                            converted_video_path = os.path.join(run_folder_dir, 'converted_video.mp4')
                            cmd = [
                            'HandBrakeCLI',
                            '-i', video_file_path,
                            '-o', converted_video_path
                            ]

                            subprocess.run(cmd)
                            video_file_path = converted_video_path  # Update the video file path to the converted video

                        audio = AudioSegment.from_file(video_file_path, format="mp4")
                        wav_file_path = os.path.splitext(self.video_file_paths[0])[0] + '.wav'
                        audio.export(wav_file_path, format="wav")

                        transcriber.generate_srt(wav_file_path, srt_path)
                        os.remove(wav_file_path)

                        if language == "Vietnamese":
                            self.translate_srt(srt_path, 'vi')   

                        cmd = [
                        'HandBrakeCLI',
                        '-i', self.video_file_paths[i],
                        '-o', output_file_name,
                        '--srt-file', srt_path,
                        '--srt-codeset', 'UTF-8',
                        '--srt-burn=1'
                        ]

                        subprocess.run(cmd)
                        print(f'Video exported successfully: {output_file_name}')
                        os.remove(srt_path)         
                        shutil.rmtree(run_folder_dir)
                finally:
                    self.task_completed.emit()
                    # Check if the user wants to open the SRT file after generation
                    if self.open_file_after_completion:
                        os.startfile(folder)
                    

            # Create and start the thread
            threading.Thread(target=task).start()

    def export_video(self):
        # Check if a file has been opened
        if not self.video_file_paths:
            # If no file has been opened, show an error message
            QMessageBox.critical(self, "Error", "You must open a video file before saving video.")
            return

        language = self.ask_language()
        if not language:
            QMessageBox.critical(self, "Error", "You must select a language.")
            return

        self.open_file_after_completion = self.ask_open_file()

        # Ask the user if they want to use their own SRT file
        if len(self.video_file_paths) == 1:
            self.export_single_video(language)
            """
            use_own_srt = QMessageBox.question(self, "Use Own SRT", "Do you want to choose your own SRT file?", QMessageBox.Yes | QMessageBox.No)

            if use_own_srt == QMessageBox.Yes:
                # Let the user choose their own SRT file
                srt_file_path, _ = QFileDialog.getOpenFileName(self, "Open SRT File", "", "SRT Files (*.srt)")
                if not srt_file_path:
                    QMessageBox.critical(self, "Error", "You must select an SRT file.")
                    return
                self.srt_file_path = srt_file_path

                # Open a file dialog to save the output video file
                output_file_name, _ = QFileDialog.getSaveFileName(self, "Save Video File", "", "MP4 Files (*.mp4)")
                if not output_file_name:
                    QMessageBox.critical(self, "Error", "You must select a file name to save the video.")
                    return

                # Start the progress bar with the total duration
                self.start_progress_bar(self.video_clip.duration)

                def task():
                    try:
                        cmd = [
                        'HandBrakeCLI',
                        '-i', self.video_file_paths[0],
                        '-o', output_file_name,
                        '--srt-file', srt_file_path,
                        '--srt-codeset', 'UTF-8',
                        '--srt-burn=1'
                        ]

                        # Set up startupinfo to hide the command window
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = subprocess.SW_HIDE

                        subprocess.run(cmd, startupinfo=startupinfo)
                        print(f'Video exported successfully: {output_file_name}')
                    finally:
                        self.task_completed.emit()
                        # Check if the user wants to open the SRT file after generation
                        if self.open_file_after_completion:
                            os.startfile(os.path.dirname(output_file_name))    
                threading.Thread(target=task).start()
            else:
                self.export_single_video(language)
        """    
        else:      
            self.export_multiple_videos(language)          
               
        

    def initUI(self):        
        
        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a layout
        layout = QVBoxLayout(central_widget)
        #layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        # Add buttons to the layout       
        #self.export_button = QPushButton("Export subtitle", self)
        self.export_video_button=QPushButton("Export Video",self)
        self.export_video_button.setFixedSize(800, 50)  # Set the size of the button
        #layout.addWidget(self.export_button)
        layout.addWidget(self.export_video_button)        

        #Create connects for Export subtitle button
        #self.export_button.clicked.connect(self.export_subtitle)

        # Add a label for displaying the progress percentage
        self.progress_label = QLabel(self)
        layout.addWidget(self.progress_label)

        #Create connects for Export video button
        self.export_video_button.clicked.connect(self.export_video)

        # Create a group box     
        self.video_box = QGroupBox("")
        self.video_box.setStyleSheet("QGroupBox { border: none; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 0; }")
        layout.addWidget(self.video_box)

        # Set the stretch factor of the group box to 1
        # This will make it take up the remaining space in the layout
        layout.setStretchFactor(self.video_box, 1)

        # Create a layout for the group box
        video_layout = QVBoxLayout(self.video_box)
    
        # Create a video widget
        self.video_widget = QVideoWidget()
        video_layout.addWidget(self.video_widget)
        
        # Create video_info_label and add it to the layout
        self.video_info_label = QLabel()
        layout.addWidget(self.video_info_label)

        # Create preview_label and add it to the layout
        self.preview_label = QLabel()
        layout.addWidget(self.preview_label)
             
        # Create a menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        editMenu = menubar.addMenu('Edit')
        helpMenu = menubar.addMenu('Help')
              
        # Add actions to the file menu
        loadAction = QAction('Open', self)
        loadAction.triggered.connect(self.load_video)
        fileMenu.addAction(loadAction)
     

        # Add actions to the edit menu
        self.generate_button= QPushButton()
        generateAction = QAction('Export Video', self)
        generateAction.triggered.connect(self.export_video)
        editMenu.addAction(generateAction)

        #Add actions to the help menu
        aboutAction = QAction('About', self)
        aboutAction.triggered.connect(self.showAboutDialog)
        helpMenu.addAction(aboutAction)

        # Create a status bar
        self.statusBar().showMessage('Ready')

        # Create a frame below the "Export Video" button
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.Box)
        self.video_frame.setFrameShadow(QFrame.Raised)
        self.video_frame.setLineWidth(0)
        self.video_frame.setFixedSize(800, 300)  # Set the size of the frame

        # Set the color of the frame
        self.video_frame.setStyleSheet("""
        QFrame {
        background-color: none;
        border: none;
        }
        """)

        # Get the current directory and construct the path to the icon
        current_directory = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_directory, 'insert.png')

        # Create a button with an icon inside the frame
        self.icon_button = QPushButton(self.video_frame)
        pixmap = QPixmap(icon_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Resize the icon
        self.icon_button.setIcon(QIcon(pixmap))  # Set the resized icon
        self.icon_button.setIconSize(pixmap.size())  # Set the icon size
        self.icon_button.setFixedSize(120, 120)  # Set the button size
        self.icon_button.setStyleSheet("border: none; margin-top: -40px;")  # Remove the button border
        self.icon_button.setCursor(Qt.PointingHandCursor)  # Change cursor to pointing hand

        # Connect the button's clicked signal to the load_video method
        self.icon_button.clicked.connect(self.load_video)

        # Create a layout for the frame and add the icon label to it
        frame_layout = QVBoxLayout(self.video_frame)
        frame_layout.addWidget(self.icon_button)
        frame_layout.setAlignment(Qt.AlignCenter)

        # Add the frame to the main layout
        layout.addWidget(self.video_frame)
      


if __name__ == "__main__":    
    #run_as_admin()  # Ensure the script is running with admin rights
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


