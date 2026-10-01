import sys
print('Python works')
sys.path.insert(0, 'src')
print('Path inserted')
try:
    from miaosuan.data.acquisition import DataAcquisition
    print('Import successful')
    acq = DataAcquisition()
    print('Instance created')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
