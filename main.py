from src.datasets.TOSCA import TOSCA
from src.datasets.transforms import FLBOTransform

if __name__ == '__main__':
    dataset = TOSCA(
        root='data/TOSCA',
        category='cat',
        pre_transform=FLBOTransform(n_angles=8, alpha=10.0, tau=0.5)
    )

    print(f"Le dataset contient {len(dataset)} formes.")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
