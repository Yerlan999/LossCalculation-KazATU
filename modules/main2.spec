# -*- mode: python -*-
from pathlib import Path

block_cipher = None


a = Analysis(['main2.py'],
             pathex=[Path('C:/Users/Sony/Desktop/University/LossCalculation-KazATU/main2.py')],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

a.datas += [('VL1.jpg',Path('pics/VL1.jpg'), "DATA"),
            ('VL2.jpg',Path('pics/VL2.jpg'), "DATA"),
            ('VL3.jpg',Path('pics/VL3.jpg'), "DATA"),
            ('VL1_mir.jpg',Path('pics/VL1_mir.jpg'), "DATA"), ]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

ic_p = Path('C:/Users/Sony/Desktop/University/LossCalculation-KazATU/icon/icon.ico')

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='KazATUProgram2',
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon=str(ic_p))
