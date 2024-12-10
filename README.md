# Predictive Modeling Olympic Medal Trend

This project focuses on predicting Olympic medal trends using machine learning techniques.
* Python version required for this project -> 3.12
## **Setup Virtual Environment**

### **Windows Users**
1. **Install Python:**  
   Ensure Python 3.6 or higher is installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/). During installation, check the box to add Python to your system PATH.

2. **Create Virtual Environment:**  
   Open Command Prompt and navigate to your project directory. Run the following command to create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. **Activate Virtual Environmen:t**  
  To activate the environment, run:
```bash
  venv\Scripts\activate
```
4. **Install Dependencies:**  
   
Install the required Python packages using pip:
  ```bash
    pip install -r requirements.txt
  ```
5. **Deactivate Virtual Environment:**
   After working on the project, deactivate the environment by running:
   ```bash
   code deactivate
   ```

### **Ubuntu Users**  

1. Install Python  
Most Ubuntu systems come with Python pre-installed. Verify the version using:
```bash
  python3 --version
```
3. **Install venv Module**  
  If venv is not already installed, use the following command:
```bash
  sudo apt install python3-venv
```
3. Create Virtual Environment  
  Navigate to your project directory and create a virtual environment:
```bash
  python3 -m venv venv
```
4. Activate Virtual Environment  
To activate the environment, run:
```bash
  source venv/bin/activate
```
5. Install Dependencies  
Install the required Python packages:
```bash
  pip install -r requirements.txt
```
6. Running Project
   ```bash
      streamlit run streamlit_deshboard.py
   ```
8. Deactivate Virtual Environment
When finished, deactivate the environment by running:
```bash
  deactivate
```

