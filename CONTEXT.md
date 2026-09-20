## Purpose of the codebase

The purpose of this codebase is to monitor the performance of this PC (nvidia-thor) for local LLM model hosting. I am testing several servers (ollama, unsloth studio with llama.cpp), but only one at a time. I want to find out which most capable models i can serve on this pc at what speed (token/s).
I am intentionally not specifying which server is running and which model is loaded - the  codebase (monitoring tool) must be able to detect it by probing running processes, services etc. 

In addition, make use of standard tools, like top, jtop, tegrastats, nvidia-smi and any other tool that makes sense. Also, logs of the llm server related process/service.

Monitoring must be added for both, unsloth with llama.cpp and ollama.

UI should be user friendly, clear so that user understands the state of the PC: gpu utilization, who uses the memory, how much of it is available, tokens/s etc. Some stats. Goal is to give a clear picture what models can run on the PC.

## current state

Currently this is a fresh repo, no code done.
During the implementation the tool must be self-sufficient (give tasks to a loaded model in order to monitor).
This codebase never changes models, but uses whatever is available/loaded.
Currently I am running unsloth via mitmweb proxy (not mandatory, can be skipped - tool is used on the same machine as model server)