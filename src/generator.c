#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int check_input(int argc)
{
	if (argc<3 || argc>3)	// If input arguments are not as much as meant to be, the algorithm ends indicating it's usage.
	{
		printf("Usage: ./generator [output_filename] [number_of_coords] \n");
		return 1;
	}
	return 0;
}

FILE *create_file(char *argv1)
{
	FILE *fp;
    char overwrite='y';
    fp =fopen(argv1,"r");	// Check if file already exists in the folder
    if (fp)
    {
		do
		{
			printf("[!] Warning: File \"%s\" already exists.\nOverwrite? (y/n): ",argv1);
			overwrite=getc(stdin);
		}while (overwrite!='y' && overwrite!='n');
		fclose(fp);
    }
    if (overwrite=='y')		// Write file either already exist or not
    {
      return fopen(argv1,"w");
    }
    return NULL;			// Do not overwrite anything
}

void write_matrix(FILE *out,unsigned int coords)
{
	int utime;
	long int ltime;
	unsigned int count,j;
	ltime=time(NULL);
	utime= (unsigned int) ltime/2;
	srand(utime);
	for (count=0;count<coords;count++)
	{
		for (j=0;j<3;j++)
		{
			fprintf(out, "%09.6f ", (float) 34*rand() / (RAND_MAX-1 ) );
		}
		fprintf(out,"\n");
	}
}

int main(int argc,char *argv[])
{
	if (check_input(argc)==1) return 1;		// Abuse test
	FILE *file = create_file(argv[1]);		// File handling
	unsigned int nocords=atoi(argv[2]);		// Number of coordinates
	if (!file)
	{
		printf("Nothing to write..\nExiting...\n");
		return 1;
	}
	printf("[-] Working...\n");
	write_matrix(file,nocords);				// Write matrix to file
	fclose(file);
	printf("[+] Done! \n" );
	return 0;
}
