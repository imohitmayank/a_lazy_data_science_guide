# Linux snippets

A collection of useful Linux (and macOS) command-line snippets for data science workflows. All commands are compatible with zsh and bash shells.

## List files by size (descending)
```sh  linenums="1"
ls -lhS
```

## Find files larger than 100MB
```sh  linenums="1"
find . -type f -size +100M
```

## Count lines in all CSV files in a directory
```sh  linenums="1"
wc -l *.csv
```

## Show top 10 memory-consuming processes
```sh  linenums="1"
ps aux --sort=-%mem | head -n 11
```

## Search for a pattern in all Python files
```sh  linenums="1"
grep -rnw . -e 'pattern' --include=*.py
```

## Replace text in multiple files (in-place)
```sh  linenums="1"
sed -i '' 's/oldtext/newtext/g' *.txt
```

## Download a file from the internet
```sh  linenums="1"
curl -O https://example.com/file.csv
```

## Extract a tar.gz archive
```sh  linenums="1"
tar -xzvf archive.tar.gz
```

## Monitor disk usage in current directory
```sh  linenums="1"
du -sh *
```

## Show GPU usage (NVIDIA)
```sh  linenums="1"
nvidia-smi
```

Sample output:

```sh linenums="1"
Wed Jun 11 12:18:15 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 PCIe               Off | 00000000:00:08.0 Off |                    0 |
| N/A   37C    P0             117W / 350W |  27085MiB / 81559MiB |     51%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H100 PCIe               Off | 00000000:00:09.0 Off |                    0 |
| N/A   38C    P0              53W / 350W |      3MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    213705      C   /usr/bin/python3                          27076MiB |
+---------------------------------------------------------------------------------------+
```

## Check Python package versions in environment
```sh  linenums="1"
pip freeze
```

## Create a virtual environment (Python 3)
```sh  linenums="1"
python3 -m venv venv
source venv/bin/activate
```

## Kill a process by name
```sh  linenums="1"
pkill -f process_name
```

## Count unique values in a CSV column
```sh  linenums="1"
cut -d, -f2 file.csv | sort | uniq -c | sort -nr
```

## Preview a CSV file (first 5 rows)
```sh  linenums="1"
head -n 5 file.csv
```

## Check open ports
```sh  linenums="1"
lsof -i -P -n | grep LISTEN
```

## Download all images from a webpage
```sh  linenums="1"
wget -nd -r -P ./images -A jpg,jpeg,png,gif http://example.com
```

## Show the 10 largest files in a directory tree
```sh  linenums="1"
find . -type f -exec du -h {} + | sort -rh | head -n 10
```

## Remove all files except .csv in a directory
```sh  linenums="1"
find . ! -name '*.csv' -type f -delete
```

## Split a large CSV into smaller files (1000 lines each)
```sh  linenums="1"
split -l 1000 bigfile.csv smallfile_
```

## Find the 10 largest directories in the current directory
```sh  linenums="1"
du -h --max-depth=1 | sort -hr | head -n 10
```

---

Feel free to copy, modify, and combine these snippets for your data science projects!
