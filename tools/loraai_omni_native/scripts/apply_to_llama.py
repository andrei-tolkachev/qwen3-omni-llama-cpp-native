#!/usr/bin/env python3
"""Install the isolated Omni module and a small precision patch in a llama checkout."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HOOK='''\n# LoraAI native Qwen3 Omni (isolated, opt-in target).
option(LLAMA_OMNI "Build native Qwen3 Omni speech tools" OFF)
if(LLAMA_OMNI)
    if(NOT LLAMA_BUILD_SERVER OR NOT BUILD_SHARED_LIBS)
        message(FATAL_ERROR "LLAMA_OMNI requires LLAMA_BUILD_SERVER=ON and BUILD_SHARED_LIBS=ON")
    endif()
    add_subdirectory(tools/loraai_omni_native)
endif()
'''

def run(*args):return subprocess.run(args,capture_output=True,text=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('llama',type=Path)
    p.add_argument('--check',action='store_true');p.add_argument('--replace',action='store_true',help='Replace a reviewed existing Omni module, retaining build/release folders')
    a=p.parse_args();llama=a.llama.resolve();target=llama/'tools/loraai_omni_native'
    assert (llama/'include/llama.h').is_file(),'Not a llama.cpp source directory'
    patch=ROOT/'patches/0001-cuda-honor-f32-precision.patch'
    forward=run('git','-C',str(llama),'apply','--check',str(patch))
    reverse=run('git','-C',str(llama),'apply','--reverse','--check',str(patch))
    if forward.returncode and reverse.returncode:raise SystemExit('CUDA precision patch needs review for this revision:\n'+forward.stderr)
    files=[f for f in ROOT.rglob('*') if f.is_file() and not any(v in {'build','releases','__pycache__','.git'} or v.startswith('build.') for v in f.relative_to(ROOT).parts[:-1])]
    for f in files:
        dest=target/f.relative_to(ROOT)
        if dest.exists() and dest.read_bytes()!=f.read_bytes() and not a.replace:
            raise SystemExit('Existing module differs: '+str(dest)+'; review before --replace')
    cmake=llama/'CMakeLists.txt';text=cmake.read_text()
    if 'add_subdirectory(tools/loraai_omni_native)' not in text and 'LLAMA_OMNI' in text:raise SystemExit('Existing LLAMA_OMNI CMake configuration needs review')
    if a.check:print('Compatible: patch applies or is already present; module copy is reviewable');return
    if not forward.returncode:
        result=run('git','-C',str(llama),'apply',str(patch))
        if result.returncode:raise SystemExit(result.stderr)
    if target!=ROOT:
        for f in files:
            dest=target/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    if 'add_subdirectory(tools/loraai_omni_native)' not in text:cmake.write_text(text+HOOK)
    print('Installed isolated Omni module. Configure with -DLLAMA_OMNI=ON and your GGML GPU backend.')

if __name__=='__main__':main()
