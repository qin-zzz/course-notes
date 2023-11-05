Credit: 5 ECTS

Professor: BALZAROTTI Davide

Exam: Yes, no documents allowed

Lab: 
Challenge1: replace the `,` in the file with `|`, but ignoring those between quotes `""`

**solution1**
```shell
sed 's-,-|-g'|sed -r 's-"([^"]*?)\|([^"]*?)"-"\1,\2"-g'
```
Makefiles are used to automate the build process of (traditionally)
executable files. Unfortunately, sometimes the programs have weird
requirements...

Challenge2. In this exercise, you have to automate the building process of
a strange application that contains a different C file
every month. For example, to build the app in January you would run:
 > gcc -c January.c -o January.o
 > gcc strange_app.c January.o -o strange_app

If instead you build it again in March, the command becomes
 > gcc -c March.c -o March.o
 > gcc strange_app.c March.o -o strange_app

Write a simple makefile to build the application.
Note that the .o file has to be re-generated only when the
corresponding .c file changed, and the strange_app binary
has to be regenerated only when either the .o file or the
strange_app.c file changed.

          -----------======  HELP ======-----------

Check out the "date" command to get the name of the current month

- compile c file and generate application based on the current date only when the source file is changed

**solution2**
```Makefile
curr_month=$(shell date +%B)
all: ${curr_month}.o strange_app

${curr_month}.o:${curr_month}.c
        gcc -c ${curr_month}.c -o ${curr_month}.o

strange_app:strange_app.c ${curr_month}.o
        gcc strange_app.c ${curr_month}.o -o strange_app
```

Challenge3. filter the commands with more than 5 ups 

**solution3**

```shell
string1="num-votes-"
string2='"command"'
while read -r line;
do
        case $line in
        *"$string2"*)
                varc=`echo "$line" | sed -r 's|(.*?)class="command">(.*?)</div>|\2|g'`
        ;;
        *"$string1"*)
                var=`echo "$line" | sed -r 's|(.*?)>(.*?)</div>|\2|g'`
        if [ "$var" -ge "5" ] ; then
                 echo "$varc"
        fi
        ;;
        * );;
        esac
done
```
or
```shell
egrep '<div class="command">.*</div>|<div class="num-votes".*</div>' | sed 's/<[^>]*>//g' | pr -2ats' ' | awk '{if ($NF >= 5) {NF = NF - 1; print $0;}}'
```

Challenge4. Your goal is to check the authentication log to spot attempts to login into
your machine (over ssh) by bruteforcing the password. Any internet-facing
machine routinely receives many of such attempts, which are typically
harmless... unless a user has a very simple password, of course :)

In this folder you find a very short snippet of a real auth log file.
To achieve your objective, you want to write e command that identifies
IP addresses from which you received at least one correct login and a
number of failed login attempts **greater than** the number of successful ones.
The output should contain, for each row, the IP, the number of failed
attempts, and the number of successful one.

For instance, if your command is in solution.sh, the command:
> cat auth.log | bash solution.sh

should print:

151.62.163.222 3 1

10.0.0.202 19 2

         -----------====== SUBMISSION ======-----------

Submit a text file containing in the first line your command
as you would write it on the shell. Your command will be tested invoking:

> cat logfile | bash yourfile

- find the ip address that tried to login but failed more times than succeeded

**solution4**

```shell
awk -F': ' '/Fail|Accept/{print $2}'| awk -F' port' '{print $1}'| awk '{print $(NF),$1}'|sort|uniq -c|awk '{if($2 == prev_ip && prev_c < $1){print $2, $1, prev_c;}; prev_ip=$2; prev_c=$1}'     
```

Challenge5. The network administrator gave you a log file containing all the connections
logged over the network on a certain day. As you are investigating an incident,
you want to know the list of the country towards which there was a connection
open at the time the problem occurred (unix timestamp 1364803829) from any
host in the 172.30.1/24 network.

You can resolve an IP to a country by running: geoiplookup <ip>

For instance, if you run your code on the log snippet you find
in this folder, the output should be:

      5  US, United States
      2  IP Address not found
      1  TW, Taiwan
      1  MY, Malaysia
      1  IN, India
      1  ES, Spain
      1  DK, Denmark
      1  CY, Cyprus
      1  CN, China
      1  CA, Canada

 -----------====== Submission  ======--------------

Submit a text file containing the commands in your solution.
Your file will be tested running:
```bash
$> cat log_file | bash yourfile
```
**solution5**
```shell
cat challenges/shell5/netflow_log | awk -F, '($1+$8)>=1364803829 {print $4,$5}'|grep 172.30.1 | sed 's/172.30.1.* //'| sed 's/172.30.1.*//'|awk 'system("geoiplookup " $1)'|sed 's/GeoIP Country Edition:/ /'|awk -F, '{a[$0]++;}END{for (i in a)print a[i],i;}'|sort -rn
``` 
or
```shell
cat challenges/shell5/netflow_log | awk -F, '($1+$8)>=1364803829 {print $4,$5}'|grep 172.30.1 |awk 'system("geoiplookup " $1)'| sed 's/172.30.1.* //'| sed 's/172.30.1.*//'|sed 's/GeoIP Country Edition:/ /'|awk -F, '{a[$0]++;}END{for (i in a)print a[i],i;}'|sort -rn
```
