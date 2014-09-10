chisel::scripts
===============


## persrun

Runs a command and/or script. 
If targets are set, execution only happens if at least one target is not persisted.

```bash
# the help message is quite enough to understand
./persrun -h

# example: this one runs the command 
./persrun -t tests/message.txt -c "echo '`date` Hello World' > tests/message.txt" -v

# exmaple: this one doesn't 
./persrun -t tests/message.txt -c "echo '`date` Hello World' > tests/message.txt" -v

# example: this one does
touch tests/message.txt
./persrun -t tests/message.txt -c "echo '`date` Hello World' > tests/message.txt" -v

```

## HOW TO MAKE THE STANFORD PARSER

The main bulk of the Stanford Parser is contained in Parser.jar



