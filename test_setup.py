"""
测试项目配置和依赖
运行此脚本检查项目是否正确设置
"""

import sys
import os

print("="*70)
print("Testing Project Setup")
print("="*70)
print()

# 1. 检查Python版本
print("1. Checking Python version...")
version = sys.version_info
print(f"   Python {version.major}.{version.minor}.{version.micro}")
if version.major < 3 or (version.major == 3 and version.minor < 8):
    print("   ❌ ERROR: Python 3.8+ required")
    sys.exit(1)
else:
    print("   ✓ Python version OK")
print()

# 2. 检查必要的目录
print("2. Checking directories...")
required_dirs = [
    'data/raw',
    'data/processed',
    'results/models',
    'plots',
    'logs',
    'src',
    'src/models'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ❌ {dir_path} - creating...")
        os.makedirs(dir_path, exist_ok=True)
print()

# 3. 检查核心文件
print("3. Checking core files...")
required_files = [
    'config.yaml',
    'main.py',
    'requirements.txt',
    'src/data_loader.py',
    'src/feature_engineering.py',
    'src/train.py',
    'src/evaluate.py'
]

all_files_exist = True
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
    else:
        print(f"   ❌ {file_path} - missing!")
        all_files_exist = False

if not all_files_exist:
    print("\n   ERROR: Some core files are missing!")
    sys.exit(1)
print()

# 4. 检查依赖包
print("4. Checking dependencies...")
dependencies = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'PyTorch',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'tqdm': 'tqdm'
}

missing_deps = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ❌ {name} - not installed")
        missing_deps.append(name)

# 可选依赖
print("\n   Optional dependencies:")
optional_deps = {
    'yfinance': 'yfinance',
    'prophet': 'prophet',
    'statsmodels': 'statsmodels',
    'ta': 'ta'
}

for module, name in optional_deps.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ⚠ {name} - not installed (optional)")

# Mamba (特殊处理)
try:
    import mamba_ssm
    print(f"   ✓ mamba-ssm")
except ImportError:
    print(f"   ⚠ mamba-ssm - not installed (will use GRU as fallback)")

print()

# 5. 测试配置文件
print("5. Testing configuration file...")
try:
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ config.yaml loaded successfully")
    print(f"   - Stock codes: {len(config['data']['stock_codes'])}")
    print(f"   - Sequence length: {config['features']['sequence_length']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
except Exception as e:
    print(f"   ❌ Error loading config.yaml: {e}")
    sys.exit(1)
print()

# 6. 测试CUDA可用性
print("6. Checking CUDA availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA version: {torch.version.cuda}")
    else:
        print(f"   ⚠ CUDA not available (will use CPU)")
except Exception as e:
    print(f"   ⚠ Could not check CUDA: {e}")
print()

# 7. 测试数据加载器
print("7. Testing data loader...")
try:
    sys.path.insert(0, 'src')
    from data_loader import StockDataLoader
    loader = StockDataLoader()
    print(f"   ✓ Data loader imported successfully")
except Exception as e:
    print(f"   ❌ Error importing data loader: {e}")
    sys.exit(1)
print()

# 8. 测试模型导入
print("8. Testing model imports...")
model_imports = {
    'MLP': 'from src.models.mlp import MLPModel',
    'LSTM': 'from src.models.lstm import LSTMModel',
    'Transformer': 'from src.models.transformer import TransformerModel',
}

for model_name, import_stmt in model_imports.items():
    try:
        exec(import_stmt)
        print(f"   ✓ {model_name} model")
    except Exception as e:
        print(f"   ❌ {model_name} model: {e}")

# Mamba (特殊处理)
try:
    from src.models.mamba_model import MambaModel
    print(f"   ✓ Mamba model")
except Exception as e:
    print(f"   ⚠ Mamba model: {e} (will use fallback)")
print()

# 总结
print("="*70)
print("Summary")
print("="*70)

if missing_deps:
    print("❌ Missing dependencies:")
    for dep in missing_deps:
        print(f"   - {dep}")
    print("\nPlease install missing dependencies:")
    print("   pip install -r requirements.txt")
    print()
    sys.exit(1)
else:
    print("✓ All required dependencies are installed!")
    print("✓ Project setup is complete!")
    print()
    print("You can now run:")
    print("   python main.py")
    print()
    print("Or for a quick test:")
    print("   python main.py --step download")
    print()

print("="*70)

