# Baisc

Credit: 5ECTS

Professor: BALZAROTTI Davide

Exam: No

Assignments: 
+ 1 quiz for malware classification
+ 2 challenges for reverse engineering
+ 3 challenges for attack analysis
+ 1 project 

# Solutions
## 1-knock-knock

From the python file, we can assume that the second parameter is where the flag is saved, in the form of pair <pos, char>. The attacker then encoded all the characters as unicode and used them as the destination ports separatedly. 

```python
from scapy.all import *; 
print("".join([ch[1] for ch in sorted([(p[TCP].window, chr(p[TCP].dport-3000)) for p in rdpcap('dump.pcapng') if TCP in p and p[TCP].window >= 8000 and p[TCP].dport-3000 > 10])]))
```

Flag : FOR2020{N0tSoStealthy} 

## 2-so_small_yet_so_painful

1. Get the description about the disk with `binwalk -b -e disk.dd`
2. Locate the block that might contain the flag with `fls -r disk.dd`
3. Get detailed info about the block(NO.27) with `istat disk.dd 27`
4. Find the blocks that started with *1f 8b* with `blkls disk.dd | hexdump -C | grep "1f 8b"` and the blocks nearby, as *1f 8b* is the magic number for **.gzip**
5. Take the dump and converted the hex to binary with an online converter, saved it as **.dat**
6. Open the **.dat** file with `zcat` command

Flag : FOR2020{BadJournalsF0rN00bs}

# 3-poor_ransomware

1. List all the proccesses presented in **dump.bz2** with `python ../volatility/vol.py -f dump linux_malfind --profile=LinuxUbuntu_4_15_0-101-generic_profilex64 | grep Pid`, to have an overview, and noticed that many possible malicious processes were web-based
2. List all the processes that contain *firefox* or *chrome* with `python ../volatility/vol.py -f dump linux_psscan --profile=LinuxUbuntu_4_15_0-101-generic_profilex64 | grep firefox`, of which the process with pid 1076, 1042, 1320 were suspicious
3. Investigate into these processes with `python ../volatility/vol.py -f dump linux_malfind --profile=LinuxUbuntu_4_15_0-101-generic_profilex64 | grep firefox`, of which the one that took the most storage was a directory `/tmp/firefox`
4. Extracte the binary and found there the ransomeware
5. Decompile it with **Ghidra** to find the master key *I'm not a noob it's too easy* in *find_and_crypt*
6. From *crypt_block* we came to know the way that the key was encrypted, therefore we used the following script to decrypt 
```python
def decrypt(pid):
    m_key = "I'm not a noob it's too easy"
    var = pid % 256
    key = [var ^ ord(i) for i in m_key]
    with open("secret.txt.crypt", "rb") as f:
        cipher = f.read()
    return "".join([chr(int(cipher[i])^key[i%0x20]) for i in range(len(cipher))])
```

Flag : FOR2020{K3ysInMem0ryAreLaaAame}

