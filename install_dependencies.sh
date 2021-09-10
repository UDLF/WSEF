# script to install WSEF dependencies

# set temporary dir to build packages
mkdir tmp
export TMPDIR=/media/Data/ws-repo/ws-experiments/tmp

# upgrade pip
pip install --upgrade pip

# install main dependencies
pip install -r requirements.txt

# install torch and torch-geometric (cpu version)
pip install torch==1.9
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip install torch-geometric

# install pyopf
git clone https://github.com/thierrypin/LibOPFcpp
git clone https://github.com/marcoscleison/PyOPF
pip install pybind11
mv LibOPFcpp/include/libopfcpp PyOPF/pyopf_native_/include
cd PyOPF
python setup.py build_ext -i
python setup.py install
cd ..

# remove tmp dirs
rm -rf LibOPFcpp
rm -rf PyOPF
rm -rf tmp
