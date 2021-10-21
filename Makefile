obj-m := killd.o 
all :
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules


clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
