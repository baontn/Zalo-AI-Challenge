from .execution_code import *
from .utils import *
from tensorflow.keras.models import load_model
!git clone https://github.com/baontn/Zalo-AI-Challenge.git


def run_model(test_dir, model_path='inceptionv3.h5'):
    model_name = model_path[:model_path.find('.')]
    print(f'Model {model_name} is running')
    model = load_model(model_path)
    x_pred, pub_test = get_Xpred(test_dir)
    create_result(x_pred, model, pub_test, model_name)
    print('Result saved')

test_dir = 'Zalo-AI-Challenge/private_test/images/'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_model(test_dir)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
