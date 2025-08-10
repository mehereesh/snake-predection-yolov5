# Snake Detection using YOLOv5

This project uses YOLOv5 for snake detection in images and videos.

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/mehereesh/snake-predection-yolov5.git
    cd snake-predection-yolov5
    ```

2. **Create a virtual environment (optional but recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Inference on Images

```bash
python detect.py --source path/to/image.jpg --weights path/to/weights.pt
