CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=gnu99 -fno-strict-aliasing -I.
LIBS = -lcurl -lm

all: shadowclaw

shadowclaw: shadowclaw.c cJSON.c
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

strip: shadowclaw
	strip shadowclaw

clean:
	rm -f shadowclaw
