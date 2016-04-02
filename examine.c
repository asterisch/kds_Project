#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define MIN_LIM 12
#define MAX_LIM 30

int check_input(int argc,char *argv[])
{
	if (argc<6 || argc>6) // If input arguments are not as much as meant to be, the algorithm ends indicating it's usage.
	{
		printf("[-]Usage: ./examine [max_collisions] [max_exec_time] [input_file] [Threads] [Processes] \nUse \"-1\": for no boundies \n");
		if (argc==2) if (!strcmp(argv[1],"--help")) printf("max_collisions: Maximum number of collisions\nmax_exec_time: Maximum execution time\ninput_file: Filename to examine\nThreads: Number of threads to use\nProcesses: Number of Processes to use.\n" );
		return 1;
	}
	return 0;
}

FILE *check_input_file(char *filename)
{
	FILE *temp = fopen(filename,"r");
	if (!temp)
	{
		printf("[!]Input file does not exist.\nExiting...\n");
		return NULL;
	}
	return temp;
} 

int main(int argc,char * argv[])
{
	if (check_input(argc,argv)==1)
	{
		return 1;
	}
	FILE *input = check_input_file(argv[3]);
	if  (!input)
	{
		return 1;
	}
	int coll=atoi(argv[1]);											//,etime=argv[2],threads=argv[4],proc=argv[5];
	fclose(input);
	printf("[+]Done! \n" );
	return 0;
}
