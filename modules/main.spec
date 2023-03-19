# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=[Path('C:/Users/Пользователь/Desktop/Yerlan/Projects/LossCalculation-KazATU/modules/main.py')],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a.datas += [('VL1.jpg',Path('pics/VL1.jpg'), "DATA"),
            ('VL2.jpg',Path('pics/VL2.jpg'), "DATA"),
            ('VL1_mir.jpg',Path('pics/VL1_mir.jpg'), "DATA"), ]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

ic_p = Path('C:/Users/Пользователь/Desktop/Yerlan/Projects/LossCalculation-KazATU/modules/icon/icon.ico')

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ic_p)
)
