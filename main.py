import customtkinter as ctk
import os
import librosa
import sounddevice as sd
from PIL import Image
import sys
import os
from customtkinter import CTkImage
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from tkinter import messagebox
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from scripts.knn import K_Nearest_Neighbors
from scripts.svm import Support_Vector_Machine
from tkinter import filedialog
import soundfile as sf

class VoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Is this your voice?")
        self.geometry("800x600")
        self.resizable(True, True)
        self.widthScreen = self.winfo_screenwidth()
        self.heightScreen = self.winfo_screenheight()
        self.main_color = "#0e9aa7"
        self.create_start_page()
        self.after(0, lambda: self.wm_state('zoomed'))

    def create_start_page(self):
        self.clear_frame()

        image = Image.open("resources/bg.png")
        background = ctk.CTkImage(light_image=image, size=(self.widthScreen, self.heightScreen))

        background_label = ctk.CTkLabel(self, image=background, text="")
        background_label.place(relwidth=1, relheight=1)

        start_button = ctk.CTkButton(self, text="Start", command=self.create_main_menu,fg_color=self.main_color)
        start_button.place(relx=0.5, rely=0.5, anchor="center")

        print("Welcome to the Voice App")

    def create_main_menu(self):
        self.clear_frame()

        title = ctk.CTkLabel(self, text="Is This Your Voice?", font=("Arial", 24))
        title.place(relx=0.5, rely=0.05, anchor="center")

        options = ["Visualize Data", "Training", "Predicting"]
        images = [
            "resources/data.png",
            "resources/training.png",
            "resources/classifying.png",
        ]
        commands = [
            self.create_visualize_data,
            self.create_training_menu,
            self.create_classify_menu,
        ]

        button_width = self.widthScreen * 0.20
        button_height = self.heightScreen * 0.35
        x_offsets = [0.2, 0.5, 0.8]  
        y_offset = 0.4  

        for i, (option, image_path, command) in enumerate(zip(options, images, commands)):
            image = ctk.CTkImage(dark_image=Image.open(image_path), size=(button_width, button_height))
            button = ctk.CTkButton(
                self,
                image=image,
                text="",
                fg_color="lightgray",
                hover_color=self.main_color,
                command=command,
            )
            button.place(relx=x_offsets[i], rely=y_offset, anchor="center")

        exit_button = ctk.CTkButton(self, text="Exit", command=self.quit, fg_color=self.main_color, hover_color="#530000")
        exit_button.place(relx=0.5, rely=0.9, anchor="center")

        print("You are in the main menu")

    def create_training_menu(self):
        self.clear_frame()

        title = ctk.CTkLabel(self, text="Training", font=("Arial", 24))
        title.place(relx=0.5, rely=0.05, anchor="center")

        options = ["KNN", "SVM"]
        images = [
            "resources/knn.png",
            "resources/svm.png",
        ]
        commands = [
            self.train_knn_model,
            self.train_svm_model,
        ]

        button_width = self.widthScreen * 0.2
        button_height = self.heightScreen * 0.3
        x_offsets = [0.35, 0.65] 
        y_offset = 0.5  #

        for i, (option, image_path, command) in enumerate(zip(options, images, commands)):
            image = ctk.CTkImage(dark_image=Image.open(image_path), size=(button_width, button_height))
            button = ctk.CTkButton(
                self,
                image=image,
                text="",
                fg_color="lightgray",
                hover_color=self.main_color,
                command=command,
            )
            button.place(relx=x_offsets[i], rely=y_offset, anchor="center")

        back_button = ctk.CTkButton(self, text="Back", command=self.create_main_menu, fg_color=self.main_color, hover_color="#530000")
        back_button.place(relx=0.5, rely=0.9, anchor="center")

        print("You are in the training menu")

    def train_knn_model(self):
        self.clear_frame()

        title = ctk.CTkLabel(self, text="Training KNN Model", font=("Arial", 24))
        title.pack(pady=10)

        back_button = ctk.CTkButton(self, text="Back", command=self.create_training_menu, fg_color=self.main_color, hover_color="#530000")
        back_button.pack(pady=10)

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True)

        canvas = ctk.CTkCanvas(main_frame, bg="gray", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar_y = ctk.CTkScrollbar(main_frame, command=canvas.yview, orientation="vertical")
        scrollbar_y.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar_y.set)

        content_frame = ctk.CTkFrame(canvas)
        canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")

        content_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)

        def resize_images():
            canvas_width = canvas.winfo_width() 
            num_columns = 4  
            column_width = canvas_width // num_columns 
            image_size = int(column_width * 0.8)  

            dataset_image = CTkImage(Image.open("resources/DatasetSplit.png"), size=(image_size, image_size))
            confusion_image = CTkImage(Image.open("resources/ConfusionMatrix.png"), size=(image_size, image_size))
            report_image = CTkImage(Image.open("resources/ClassificationReport.png"), size=(image_size, image_size))
            accuracy_image = CTkImage(Image.open("resources/accuracy.png"), size=(image_size, image_size))

            dataset_info.configure(image=dataset_image)
            dataset_info.image = dataset_image

            confusion_title.configure(image=confusion_image)
            confusion_title.image = confusion_image

            classification_report_label.configure(image=report_image)
            classification_report_label.image = report_image

            accuracy_label.configure(image=accuracy_image)
            accuracy_label.image = accuracy_image

        def update_canvas_position(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
            resize_images() 

        canvas.bind("<Configure>", update_canvas_position)

        parameters_title = ctk.CTkLabel(content_frame, text="Parameters", font=("Arial", 18))
        parameters_title.grid(row=0, column=0, columnspan=4, pady=10, sticky="snew")

        neighbors_label = ctk.CTkLabel(content_frame, text="Number of Neighbors")
        neighbors_label.grid(row=1, column=0, padx=5, pady=10, sticky="e")
        neighbors_input = ctk.CTkEntry(content_frame)
        neighbors_input.insert(0, "4")
        neighbors_input.grid(row=1, column=1, padx=5, pady=10, sticky="w")

        test_size_label = ctk.CTkLabel(content_frame, text="Test Size (0-1)")
        test_size_label.grid(row=1, column=2, padx=5, pady=10, sticky="e")
        test_size_input = ctk.CTkEntry(content_frame)
        test_size_input.insert(0, "0.2")
        test_size_input.grid(row=1, column=3, padx=5, pady=10, sticky="w")

        features_label = ctk.CTkLabel(content_frame, text="Select Features", font=("Arial", 16))
        features_label.grid(row=2, column=0, columnspan=4, pady=20, sticky="snew")

        descriptors = [
            "Mel-Frequency Cepstal Coefficients", "Centroid", "Spread", "Skewness",
            "Kurtosis", "Slope", "Decrease", "Roll-off", "Flux (mean)", "Flux (variance)",
            "MPEG7_Centroid", "MPEG7_Spread", "MPEG7_Flatness", "Pitch",
            "Zero-Crossing Rate", "Log-Attack Time", "AM Index"
        ]

        selected_features = []

        def toggle_feature(self, button, descriptor):
            if descriptor in selected_features:
                selected_features.remove(descriptor)
                button.configure(fg_color="#000000", hover_color=self.main_color, text_color="white")
            else:
                selected_features.append(descriptor)
                button.configure(fg_color="white", hover_color=self.main_color, text_color="black")

        for index, descriptor in enumerate(descriptors):
            feature_button = ctk.CTkButton(
                content_frame,
                text=descriptor,
                fg_color="#000000",
                hover_color=self.main_color,
                command=lambda d=descriptor, b=None: toggle_feature(self, b, d),
            )
            feature_button.grid(row=3 + index // 4, column=index % 4, padx=5, pady=15, sticky="ew")

            feature_button.configure(command=lambda b=feature_button, d=descriptor: toggle_feature(self, b, d))
        
        def save_model():
            if hasattr(self, "knn"):
                features_str = "_".join(selected_features)
                model_name = f"knn_{features_str}_{neighbors_input.get()}_{test_size_input.get()}.joblib"

                self.knn.save_model(model_name)

                messagebox.showinfo("Model Saved", f"Model saved as {model_name}")
                print(f"Model saved as {model_name}")
            else:
                messagebox.showerror("Error", "No trained model available to save.")

        train_button = ctk.CTkButton(content_frame, text="Train", command=lambda: display_results(selected_features, neighbors_input.get(), test_size_input.get()), fg_color=self.main_color)
        train_button.grid(row=10, column=1, pady=15, sticky="ew")

        self.save_button = ctk.CTkButton(content_frame, text="Save", command=save_model, state="disabled", fg_color=self.main_color)
        self.save_button.grid(row=10, column=2, pady=15, sticky="ew")

        dataset_info = ctk.CTkLabel(content_frame, text="Dataset Split", font=("Arial", 18), compound="top")
        dataset_info.grid(row=11, column=0, pady=10, sticky="ew")

        confusion_title = ctk.CTkLabel(content_frame, text="Confusion Matrix", font=("Arial", 18), compound="top")
        confusion_title.grid(row=11, column=1, pady=10, sticky="ew")

        classification_report_label = ctk.CTkLabel(content_frame, text="Classification Report", font=("Arial", 18), compound="top")
        classification_report_label.grid(row=11, column=2, pady=10, sticky="ew")

        accuracy_label = ctk.CTkLabel(content_frame, text="Accuracy", font=("Arial", 18), compound="top")
        accuracy_label.grid(row=11, column=3, pady=10, sticky="ew")
    
        def update_image(widget, image):
            try:
                if hasattr(widget, "image"):
                    del widget.image  
                widget.configure(image=image)
                widget.image = image
            except Exception as e:
                print(f"Error updating image for {widget}: {e}")

        def display_results(list_of_descriptors, n_neighbors, test_size):
            try:
                n_neighbors = int(n_neighbors)
                test_size = float(test_size)
                messagebox.showinfo("Training", "Training the KNN model. This may take a few seconds...")
                data_path = "Data"
                self.knn = K_Nearest_Neighbors(n_neighbors)  
                self.knn.create_dataset(data_path, list_of_descriptors)
                self.knn.split_dataset(test_size)
                self.knn.train()
                self.knn.evaluate()

                canvas_width = canvas.winfo_width() 
                num_columns = 4  
                column_width = canvas_width // num_columns  
                image_size = int(column_width * 0.8)  

                train_counts = pd.Series(self.knn.y_train).value_counts()
                test_counts = pd.Series(self.knn.y_test).value_counts()

                all_classes = sorted(set(train_counts.index) | set(test_counts.index))
                data = {
                    'Training': [train_counts.get(cls, 0) for cls in all_classes],
                    'Testing': [test_counts.get(cls, 0) for cls in all_classes],
                }

                df = pd.DataFrame(data, index=all_classes)
                df.loc['Total'] = df.sum()
                df['Total'] = df.sum(axis=1)

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis('tight')
                ax.axis('off')

                table = ax.table(cellText=df.values,
                                rowLabels=df.index,
                                colLabels=df.columns,
                                cellLoc='center',
                                loc='center',
                                colWidths=[0.2]*len(df.columns))

                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)

                for i in range(len(df.index) + 1):
                    for j in range(len(df.columns)):
                        cell = table[(i, j)]
                        if i == 0: 
                            cell.set_facecolor('#4e8e99')
                            cell.set_text_props(color='white')
                        elif i == len(df.index):  
                            cell.set_facecolor('#e6e6e6')
                        elif j == len(df.columns) - 1:  
                            cell.set_facecolor('#e6e6e6')

                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                buf.seek(0)
                dataset_image = CTkImage(Image.open(buf), size=(image_size, image_size))

                accuracy_text = f"Accuracy: {self.knn.accuracy:.2f}"
                accuracy_label.configure(text=accuracy_text)

                acc = self.knn.accuracy
                if acc >= 0.7:
                    imgPathAcc = "resources/good.png"
                elif acc >= 0.35:
                    imgPathAcc = "resources/ok.png"
                else:
                    imgPathAcc = "resources/bad.png"

                accuracy_image = CTkImage(light_image=Image.open(imgPathAcc), size=(image_size, image_size))

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.matshow(self.knn.confusionMatrix, cmap=plt.cm.Blues, alpha=0.7)
                for i in range(self.knn.confusionMatrix.shape[0]):
                    for j in range(self.knn.confusionMatrix.shape[1]):
                        ax.text(x=j, y=i, s=self.knn.confusionMatrix[i, j], ha='center', va='center')
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")

                buf2 = BytesIO()
                plt.savefig(buf2, format="png")
                buf2.seek(0)
                confusion_image = CTkImage(Image.open(buf2), size=(image_size, image_size))

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.axis("off")
                ax.text(0.01, 0.99, self.knn.classificationReport, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top', family="monospace")
                plt.tight_layout()

                buf3 = BytesIO()
                plt.savefig(buf3, format="png")
                buf3.seek(0)
                report_image = CTkImage(Image.open(buf3), size=(image_size, image_size))

                update_image(dataset_info, dataset_image)
                update_image(confusion_title, confusion_image)
                update_image(classification_report_label, report_image)
                update_image(accuracy_label, accuracy_image)

                self.save_button.configure(state="normal")

                buf.close()
                buf2.close()
                buf3.close()

                print("Model trained successfully")
                print(f"Descriptors: {list_of_descriptors}")
                print(f"Number of Neighbors: {n_neighbors}")
                print(f"Test Size: {test_size}")
                print(f"Accuracy: {self.knn.accuracy:.2f}")
                print(f"Confusion Matrix:\n{self.knn.confusionMatrix}")
                print(f"Classification Report:\n{self.knn.classificationReport}")

            except ValueError as ve:
                messagebox.showerror("Parameter Error", f"Invalid parameters: {ve}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        content_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        print("You are training a KNN model")

    def train_svm_model(self):
        self.clear_frame()

        title = ctk.CTkLabel(self, text="Training SVM Model", font=("Arial", 24))
        title.pack(pady=10)

        back_button = ctk.CTkButton(self, text="Back", command=self.create_training_menu, fg_color=self.main_color, hover_color="#530000")
        back_button.pack(pady=10)

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True)

        canvas = ctk.CTkCanvas(main_frame, bg="gray", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar_y = ctk.CTkScrollbar(main_frame, command=canvas.yview, orientation="vertical")
        scrollbar_y.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar_y.set)

        content_frame = ctk.CTkFrame(canvas)
        canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")

        content_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)

        def resize_images():
            canvas_width = canvas.winfo_width()
            num_columns = 4
            column_width = canvas_width // num_columns
            image_size = int(column_width * 0.8)

            dataset_image = CTkImage(Image.open("resources/DatasetSplit.png"), size=(image_size, image_size))
            confusion_image = CTkImage(Image.open("resources/ConfusionMatrix.png"), size=(image_size, image_size))
            report_image = CTkImage(Image.open("resources/ClassificationReport.png"), size=(image_size, image_size))
            accuracy_image = CTkImage(Image.open("resources/accuracy.png"), size=(image_size, image_size))

            dataset_info.configure(image=dataset_image)
            dataset_info.image = dataset_image

            confusion_title.configure(image=confusion_image)
            confusion_title.image = confusion_image

            classification_report_label.configure(image=report_image)
            classification_report_label.image = report_image

            accuracy_label.configure(image=accuracy_image)
            accuracy_label.image = accuracy_image

        def update_canvas_position(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
            resize_images()

        canvas.bind("<Configure>", update_canvas_position)

        parameters_title = ctk.CTkLabel(content_frame, text="Parameters", font=("Arial", 18))
        parameters_title.grid(row=0, column=0, columnspan=4, pady=10, sticky="snew")

        kernel_label = ctk.CTkLabel(content_frame, text="Kernel")
        kernel_label.grid(row=1, column=0, padx=5, pady=10, sticky="e")
        kernel_input = ctk.CTkEntry(content_frame)
        kernel_input.insert(0, "rbf")
        kernel_input.grid(row=1, column=1, padx=5, pady=10, sticky="w")

        C_label = ctk.CTkLabel(content_frame, text="C")
        C_label.grid(row=1, column=2, padx=5, pady=10, sticky="e")
        C_input = ctk.CTkEntry(content_frame)
        C_input.insert(0, "1.0")
        C_input.grid(row=1, column=3, padx=5, pady=10, sticky="w")

        gamma_label = ctk.CTkLabel(content_frame, text="Gamma")
        gamma_label.grid(row=2, column=0, padx=5, pady=10, sticky="e")
        gamma_input = ctk.CTkEntry(content_frame)
        gamma_input.insert(0, "scale")
        gamma_input.grid(row=2, column=1, padx=5, pady=10, sticky="w")

        test_size_label = ctk.CTkLabel(content_frame, text="Test Size (0-1)")
        test_size_label.grid(row=2, column=2, padx=5, pady=10, sticky="e")
        test_size_input = ctk.CTkEntry(content_frame)
        test_size_input.insert(0, "0.2")
        test_size_input.grid(row=2, column=3, padx=5, pady=10, sticky="w")

        features_label = ctk.CTkLabel(content_frame, text="Select Features", font=("Arial", 16))
        features_label.grid(row=3, column=0, columnspan=4, pady=20, sticky="snew")

        descriptors = [
            "Mel-Frequency Cepstal Coefficients", "Centroid", "Spread", "Skewness",
            "Kurtosis", "Slope", "Decrease", "Roll-off", "Flux (mean)", "Flux (variance)",
            "MPEG7_Centroid", "MPEG7_Spread", "MPEG7_Flatness", "Pitch",
            "Zero-Crossing Rate", "Log-Attack Time", "AM Index"
        ]

        selected_features = []

        def toggle_feature(self, button, descriptor):
            if descriptor in selected_features:
                selected_features.remove(descriptor)
                button.configure(fg_color="#000000", hover_color=self.main_color, text_color="white")
            else:
                selected_features.append(descriptor)
                button.configure(fg_color="white", hover_color=self.main_color, text_color="black")

        for index, descriptor in enumerate(descriptors):
            feature_button = ctk.CTkButton(
                content_frame,
                text=descriptor,
                fg_color="#000000",
                hover_color=self.main_color,
                command=lambda d=descriptor, b=None: toggle_feature(self, b, d),
            )
            feature_button.grid(row=4 + index // 4, column=index % 4, padx=5, pady=15, sticky="ew")

            feature_button.configure(command=lambda b=feature_button, d=descriptor: toggle_feature(self, b, d))

        def save_model():
            if hasattr(self, "svm"):
                features_str = "_".join(selected_features)
                model_name = f"svm_{features_str}_{kernel_input.get()}_{C_input.get()}_{gamma_input.get()}_{test_size_input.get()}.joblib"
                self.svm.save_model(model_name)
                messagebox.showinfo("Model Saved", f"Model saved as {model_name}")
                print(f"Model saved as {model_name}")
            else:
                messagebox.showerror("Error", "No trained model available to save.")

        train_button = ctk.CTkButton(content_frame, text="Train", command=lambda: display_results(selected_features, kernel_input.get(), C_input.get(), gamma_input.get(), test_size_input.get()), fg_color=self.main_color)
        train_button.grid(row=10, column=1, pady=15, sticky="ew")

        self.save_button = ctk.CTkButton(content_frame, text="Save", command=save_model, state="disabled", fg_color=self.main_color)
        self.save_button.grid(row=10, column=2, pady=15, sticky="ew")

        dataset_info = ctk.CTkLabel(content_frame, text="Dataset Split", font=("Arial", 18), compound="top")
        dataset_info.grid(row=11, column=0, pady=10, sticky="ew")

        confusion_title = ctk.CTkLabel(content_frame, text="Confusion Matrix", font=("Arial", 18), compound="top")
        confusion_title.grid(row=11, column=1, pady=10, sticky="ew")

        classification_report_label = ctk.CTkLabel(content_frame, text="Classification Report", font=("Arial", 18), compound="top")
        classification_report_label.grid(row=11, column=2, pady=10, sticky="ew")

        accuracy_label = ctk.CTkLabel(content_frame, text="Accuracy", font=("Arial", 18), compound="top")
        accuracy_label.grid(row=11, column=3, pady=10, sticky="ew")

        def update_image(widget, image):
            try:
                if hasattr(widget, "image"):
                    del widget.image
                widget.configure(image=image)
                widget.image = image
            except Exception as e:
                print(f"Error al actualizar imagen para {widget}: {e}")

        def display_results(list_of_descriptors, kernel, C, gamma, test_size):
            try:
                C = float(C)
                test_size = float(test_size)

                valid_kernels = ["linear", "poly", "rbf", "sigmoid"]
                if kernel not in valid_kernels:
                    raise ValueError(f"Invalid kernel: {kernel}")
                
                valid_gammas = ["scale", "auto"]
                if gamma not in valid_gammas:
                    raise ValueError(f"Invalid gamma: {gamma}")
                
                messagebox.showinfo("Training", "Training the SVM model. This may take a few seconds...")
                data_path = "Data"
                self.svm = Support_Vector_Machine(kernel, C, gamma)  
                self.svm.create_dataset(data_path, list_of_descriptors)
                self.svm.split_dataset(test_size)
                self.svm.train()
                self.svm.evaluate()

                canvas_width = canvas.winfo_width()
                num_columns = 4
                column_width = canvas_width // num_columns
                image_size = int(column_width * 0.8)

                train_counts = pd.Series(self.svm.y_train).value_counts()
                test_counts = pd.Series(self.svm.y_test).value_counts()

                all_classes = sorted(set(train_counts.index) | set(test_counts.index))
                data = {
                    'Training': [train_counts.get(cls, 0) for cls in all_classes],
                    'Testing': [test_counts.get(cls, 0) for cls in all_classes],
                }

                df = pd.DataFrame(data, index=all_classes)
                df.loc['Total'] = df.sum()
                df['Total'] = df.sum(axis=1)

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis('tight')
                ax.axis('off')

                table = ax.table(cellText=df.values,
                                rowLabels=df.index,
                                colLabels=df.columns,
                                cellLoc='center',
                                loc='center',
                                colWidths=[0.2]*len(df.columns))

                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)

                for i in range(len(df.index) + 1):
                    for j in range(len(df.columns)):
                        cell = table[(i, j)]
                        if i == 0: 
                            cell.set_facecolor('#4e8e99')
                            cell.set_text_props(color='white')
                        elif i == len(df.index):  
                            cell.set_facecolor('#e6e6e6')
                        elif j == len(df.columns) - 1:  
                            cell.set_facecolor('#e6e6e6')

                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                buf.seek(0)
                dataset_image = CTkImage(Image.open(buf), size=(image_size, image_size))

                accuracy_text = f"Accuracy: {self.svm.accuracy:.2f}"
                accuracy_label.configure(text=accuracy_text)

                acc = self.svm.accuracy
                if acc >= 0.7:
                    imgPathAcc = "resources/good.png"
                elif acc >= 0.35:
                    imgPathAcc = "resources/ok.png"
                else:
                    imgPathAcc = "resources/bad.png"

                accuracy_image = CTkImage(Image.open(imgPathAcc), size=(image_size, image_size))

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.matshow(self.svm.confusionMatrix, cmap=plt.cm.Blues, alpha=0.7)
                for i in range(self.svm.confusionMatrix.shape[0]):
                    for j in range(self.svm.confusionMatrix.shape[1]):
                        ax.text(x=j, y=i, s=self.svm.confusionMatrix[i, j], ha='center', va='center')
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")

                buf2 = BytesIO()
                plt.savefig(buf2, format="png")
                buf2.seek(0)
                confusion_image = CTkImage(Image.open(buf2), size=(image_size, image_size))

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.axis("off")
                ax.text(0.01, 0.99, self.svm.classificationReport, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top', family="monospace")
                plt.tight_layout()

                buf3 = BytesIO()
                plt.savefig(buf3, format="png")
                buf3.seek(0)
                report_image = CTkImage(Image.open(buf3), size=(image_size, image_size))

                update_image(dataset_info, dataset_image)
                update_image(confusion_title, confusion_image)
                update_image(classification_report_label, report_image)
                update_image(accuracy_label, accuracy_image)

                self.save_button.configure(state="normal")

                buf.close()
                buf2.close()
                buf3.close()

                print("Model trained successfully")
                print(f"Descriptors: {list_of_descriptors}")
                print(f"Kernel: {kernel}")
                print(f"C: {C}")
                print(f"Gamma: {gamma}")
                print(f"Test Size: {test_size}")
                print(f"Accuracy: {self.svm.accuracy:.2f}")
                print(f"Confusion Matrix:\n{self.svm.confusionMatrix}")
                print(f"Classification Report:\n{self.svm.classificationReport}")

            except ValueError as ve:
                messagebox.showerror("Parameter Error", f"Invalid parameters: {ve}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        content_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        print("You are training a SVM model")

    def create_classify_menu(self):
        self.clear_frame()

        self.current_audio_path = None
        
        title = ctk.CTkLabel(self, text="Classification", font=("Arial", 24))
        title.place(relx=0.5, rely=0.05, anchor="center")

        main_frame = ctk.CTkFrame(self)
        main_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.7)

        def choose_model():
            file_path = filedialog.askopenfilename(
                title="Select Model",
                filetypes=[("Joblib files", "*.joblib")],
                initialdir="Models"
            )
            if file_path:
                model_name = os.path.basename(file_path)
                if model_name.startswith('knn'):
                    self.current_model = K_Nearest_Neighbors()
                else:
                    self.current_model = Support_Vector_Machine()
                
                self.current_model.load_model(model_name)
                model_info.configure(text=f"Model loaded: {model_name}")
                enable_buttons()

        model_button = ctk.CTkButton(
            main_frame,
            text="Choose Model",
            command=choose_model,
            fg_color=self.main_color
        )
        model_button.place(relx=0.5, rely=0.1, anchor="center")

        model_info = ctk.CTkLabel(
            main_frame,
            text="No model loaded",
            font=("Arial", 16)
        )
        model_info.place(relx=0.5, rely=0.2, anchor="center")

        audio_frame = ctk.CTkFrame(main_frame)
        audio_frame.place(relx=0.5, rely=0.4, anchor="center")

        def record_audio():
            try:
                fs = 44100
                duration = 2  
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                record_button.configure(text="Recording...", state="disabled")
                self.update()
                sd.wait()
                record_button.configure(text="Record Audio", state="normal")
         
                temp_path = "resources/tempData/temp_recording.wav"
                self.current_audio_path = temp_path
                sf.write(temp_path, recording, fs)
 
                classify_audio(temp_path)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error recording audio: {str(e)}")

        def choose_audio():
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[("WAV files", "*.wav")]
            )
            if file_path:
                self.current_audio_path = file_path
                classify_audio(file_path)

        def extract_descriptors(model_name):
            available_descriptors = [
                "Mel-Frequency Cepstal Coefficients", "Centroid", "Spread", "Skewness",
                "Kurtosis", "Slope", "Decrease", "Roll-off", "Flux (mean)", "Flux (variance)",
                "MPEG7_Centroid", "MPEG7_Spread", "MPEG7_Flatness", "Pitch",
                "Zero-Crossing Rate", "Log-Attack Time", "AM Index"
            ]

            parts = model_name.split('_')
            
            found_descriptors = []
            current_descriptor = []

            if model_name.startswith('knn'):
                relevant_parts = parts[1:-2] 
            else:
                relevant_parts = parts[1:-4]  

            i = 0
            while i < len(relevant_parts):
                for j in range(i + 1, len(relevant_parts) + 1):
                    potential_descriptor = '_'.join(relevant_parts[i:j])
                    if potential_descriptor in available_descriptors:
                        found_descriptors.append(potential_descriptor)
                        i = j
                        break
                else:
                    i += 1
                    
            return found_descriptors
        
        def play_current_audio(self):
            if hasattr(self, 'current_audio_path') and self.current_audio_path:
                try:
                    audio, sr = librosa.load(self.current_audio_path, sr=44100)
                    sd.play(audio, sr)
                    sd.wait()
                except Exception as e:
                    messagebox.showerror("Error", f"Error playing audio: {str(e)}")
        
        def classify_audio(audio_path):
            try:
                if hasattr(self, 'current_model'):
                    model_name = self.current_model.model_name
                    descriptors = extract_descriptors(model_name)
    
                    # Clasificar
                    result = self.current_model.classify(audio_path, descriptors)
                    
                    # Actualizar imagen y texto
                    image_path = get_class_image(result)
                    if image_path:
                        result_image = ctk.CTkImage(
                            light_image=Image.open(image_path),
                            dark_image=Image.open(image_path), 
                            size=(image_size, image_size)
                        )
                        result_image_button.configure(
                            image=result_image,
                            state="normal" 
                        )
                        result_image_button.image = result_image  

                    result_label.configure(text=f"Predicted: {result}")

                    print(f"Audio {audio_path} classified as {result} using model {model_name}")
                else:
                    messagebox.showerror("Error", "No model loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Error classifying audio: {str(e)}")
        
        def get_class_image(prediction):
            class_name = prediction.split('_')[0].lower()

            image_paths = {
                'casa': "resources/house.png",
                'sol': "resources/sun.png",
                'uno': "resources/one.png",
                'pan': "resources/bread.png",
                'hola': "resources/hello.png"
            }
            
            return image_paths.get(class_name, None)

        record_button = ctk.CTkButton(
            audio_frame,
            text="Record Audio",
            command=record_audio,
            state="disabled",
            fg_color=self.main_color
        )
        record_button.grid(row=0, column=0, padx=10)

        choose_audio_button = ctk.CTkButton(
            audio_frame,
            text="Choose Audio",
            command=choose_audio,
            state="disabled",
            fg_color=self.main_color
        )
        choose_audio_button.grid(row=0, column=1, padx=10)

        result_frame = ctk.CTkFrame(main_frame)
        result_frame.place(relx=0.5, rely=0.75, anchor="center")

        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_columnconfigure(1, weight=1)

        image_size = int(self.winfo_screenwidth() * 0.10)

        result_image_button = ctk.CTkButton(
            result_frame,
            text="",
            image=None,
            command=lambda: play_current_audio(self), 
            state="disabled", 
            fg_color="transparent",  
            hover_color=self.main_color, 
            width=image_size, 
            height=image_size 
        )
        result_image_button.grid(row=0, column=0, padx=20, pady=10)

        result_label = ctk.CTkLabel(
            result_frame,
            text="Result: None",
            font=("Arial", 18)
        )
        result_label.grid(row=0, column=1, padx=20, pady=10)

        def enable_buttons():
            record_button.configure(state="normal")
            choose_audio_button.configure(state="normal")

        back_button = ctk.CTkButton(
            self,
            text="Back",
            command=self.create_main_menu,
            fg_color=self.main_color,
            hover_color="#530000"
        )
        back_button.place(relx=0.5, rely=0.9, anchor="center")

        print("You are in the classification menu")

    def create_visualize_data(self):
        self.clear_frame()

        title = ctk.CTkLabel(self, text="Visualize Data", font=("Arial", 24))
        title.place(relx=0.5, rely=0.05, anchor="center")

        names = ["Alvaro", "Carlos", "Angel", "Navil"]
        images = [
            "resources/man.png",
            "resources/man.png",
            "resources/man.png",
            "resources/woman.png",
        ]

        button_width = self.widthScreen * 0.24
        button_height = self.heightScreen * 0.24
        x_offsets = [0.25, 0.75]  
        y_offsets = [0.3, 0.6]  

        for i, (name, image_path) in enumerate(zip(names, images)):
            row, col = divmod(i, 2)
            label_offset = -0.2 if row == 0 else 0.2
            label = ctk.CTkLabel(self, text=name, font=("Arial", 18), fg_color="transparent")
            label.place(relx=x_offsets[col], rely=y_offsets[row] + label_offset, anchor="center")

            image = ctk.CTkImage(dark_image=Image.open(image_path), size=(button_width, button_height))
            button = ctk.CTkButton(
                self,
                image=image,
                text="",
                fg_color="lightgray",
                command=lambda n=name: self.create_category_view(n),
                hover_color=self.main_color,
            )
            button.place(relx=x_offsets[col], rely=y_offsets[row], anchor="center")

        back_button = ctk.CTkButton(self, text="Back", command=self.create_main_menu, fg_color=self.main_color, hover_color="#530000")
        back_button.place(relx=0.5, rely=0.9, anchor="center")

        print("You are in the visualize data menu")

    def create_category_view(self, name):
        self.clear_frame()

        title = ctk.CTkLabel(self, text=f"Audios - {name}", font=("Arial", 24))
        title.place(relx=0.5, rely=0.05, anchor="center")

        frame = ctk.CTkFrame(self, width=self.widthScreen, height=self.heightScreen * 0.85)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        canvas = ctk.CTkCanvas(frame, width=self.widthScreen * 0.95, height=self.heightScreen * 0.8, bg="black")
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ctk.CTkScrollbar(frame, orientation="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        scrollable_frame = ctk.CTkFrame(canvas, width=self.widthScreen * 0.95)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scrollbar.set)

        total_width = self.widthScreen * 0.95
        categories = ["Casa", "Hola", "Pan", "Sol", "Uno"]
        button_width = total_width / len(categories)

        for col, category in enumerate(categories):
            audio_path = f"Data/{name}/{category}/"
            if os.path.exists(audio_path):
                audio_files = [f for f in os.listdir(audio_path) if f.endswith(".wav")]

                category_label = ctk.CTkLabel(scrollable_frame, text=category, font=("Arial", 20), anchor="center")
                category_label.grid(row=0, column=col, padx=2, pady=10, sticky="nsew")

                for row, audio_file in enumerate(audio_files):
                    play_button = ctk.CTkButton(
                        scrollable_frame,
                        text=audio_file,
                        width=button_width,
                        command=lambda f=os.path.join(audio_path, audio_file): self.play_audio(f),
                        fg_color="#111311",
                        hover_color=self.main_color,
                    )
                    play_button.grid(row=row + 1, column=col, padx=2, pady=5, sticky="w")

        for col in range(len(categories)):
            scrollable_frame.grid_columnconfigure(col, weight=1)

        back_button = ctk.CTkButton(self, text="Back", command=self.create_visualize_data, fg_color=self.main_color, hover_color="#530000")
        back_button.place(relx=0.5, rely=0.95, anchor="center")

        print(f"You are viewing the category view for {name}")

    def play_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=44100)
        sd.play(audio, sr)
        sd.wait()

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    app = VoiceApp()
    app.mainloop()