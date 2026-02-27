CC = cc
CFLAGS = -O2 -Wall -Wextra -fno-strict-aliasing -I.
LDLIBS = -lcurl -lm

# cJSON is included as source
OBJS = shadowclaw.o cJSON.o

shadowclaw: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

shadowclaw.o: shadowclaw.c cJSON.h
cJSON.o: cJSON.c cJSON.h

clean:
	rm -f $(OBJS) shadowclaw

strip: shadowclaw
	strip --strip-all shadowclaw