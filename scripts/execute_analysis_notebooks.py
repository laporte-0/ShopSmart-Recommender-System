"""Execute V2 analysis notebooks in-process, without opening kernel sockets.

Only use with trusted project notebooks. Each gets a fresh IPython namespace.
This allows reproducible notebook execution in restricted WSL/sandbox sessions.
"""
import os
import argparse
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'.cache/ipython'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rag', action='store_true', help='Execute only notebook 09, preserving ML notebook outputs')
    args = parser.parse_args()
    os.chdir(ROOT)
    pattern = '09_RAG_Demo.ipynb' if args.rag else '0[6-8]*.ipynb'
    for path in sorted((ROOT/'notebooks').glob(pattern)):
        nb=nbformat.read(path,as_version=4)
        shell=InteractiveShell.instance()
        shell.reset(new_session=True)
        shell.run_cell('from IPython.display import display')
        count=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=False)
            result.raise_error()
            outputs=[]
            if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for output in captured.outputs:
                outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
            cell.outputs=outputs;cell.execution_count=count
        nb.metadata['execution_note']='All code cells executed in a fresh in-process IPython namespace. Socket-based nbclient unavailable in sandbox.'
        nbformat.validate(nb)
        nbformat.write(nb,path)
        print('Executed and validated',path.name,flush=True)

if __name__=='__main__':main()
