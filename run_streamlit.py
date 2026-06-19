import os
import sys
import streamlit.web.cli as stcli

if __name__ == '__main__':
    # Ensure current directory is in system path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    os.chdir(current_dir)
    
    # Configure arguments for Streamlit entrypoint
    sys.argv = ["streamlit", "run", "app.py", "--global.developmentMode=false"]
    sys.exit(stcli.main())
