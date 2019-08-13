import os
import sys
# add current directory
sys.path.append('./')

file_names = ['repeat_kron', 'indicator', 'random_intercepts']

for name in file_names:
    glb = {}
    exec('import ' + name, glb)
    exec('ok = '+ name + '.' + name + '()',glb)
    ok = glb['ok']
    if ok:
        print(name + ': ok')
    else:
        print(name + ': failed')
