import time
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from optiland.samples.objectives import DoubleGauss
import os


# --- 1. 定义神经网络 ---
class RayTracerSurrogate(nn.Module):
    def __init__(self):
        super(RayTracerSurrogate, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)


# --- 2. 初始化应用 ---
app = Flask(__name__, static_folder='static', template_folder='templates')

# 加载透镜系统 (带错误处理)
try:
    lens = DoubleGauss()
except Exception as e:
    lens = None
    print(f"OptiLand初始化警告: {str(e)}")

MAX_IMG_HEIGHT = 24.72

# 加载模型 (带路径检查)
device = torch.device('cpu')
model = RayTracerSurrogate().to(device)
model_path = os.path.join(os.path.dirname(__file__), 'ray_tracer_surrogate.pth')
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"模型加载失败: {str(e)}")


# --- 3. 工具函数 ---
def normalize(x, max_val=MAX_IMG_HEIGHT):
    return x / max_val


def denormalize(x, max_val=MAX_IMG_HEIGHT):
    return x * max_val


# --- 4. 路由定义 ---
@app.route('/predict_ray', methods=['POST'])
def predict_ray():
    if lens is None:
        return jsonify({"error": "OptiLand未初始化"}), 500

    data = request.get_json()
    try:
        Hx, Hy, Px, Py = data['Hx'], data['Hy'], data['Px'], data['Py']
    except KeyError:
        return jsonify({"error": "缺少必要参数"}), 400

    # 代理模型预测
    start_surrogate = time.perf_counter()
    inputs = torch.tensor([[Hx, Hy, Px, Py]], dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = denormalize(model(inputs).cpu().numpy()[0])
    time_surrogate = (time.perf_counter() - start_surrogate) * 1000

    # 解析模型预测
    start_analytical = time.perf_counter()
    wavelength = 0.5876
    rays_out = lens.trace_generic(np.array([Hx]), np.array([Hy]),
                                  np.array([Px]), np.array([Py]),
                                  np.array([wavelength]))
    analytical_result = np.array([rays_out.x[0], rays_out.y[0]])
    time_analytical = (time.perf_counter() - start_analytical) * 1000

    return jsonify({
        "surrogate_coords": {"x": float(predictions[0]), "y": float(predictions[1])},
        "analytical_coords": {"x": float(analytical_result[0]), "y": float(analytical_result[1])},
        "time_surrogate": time_surrogate,
        "time_analytical": time_analytical,
        "error": float(np.linalg.norm(predictions - analytical_result))
    })


@app.route('/health')
def health_check():
    status = {
        "model_loaded": os.path.exists(model_path),
        "optiland_ready": lens is not None
    }
    return jsonify(status), 200


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)