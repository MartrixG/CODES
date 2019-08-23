#include<cstdio>
int main()
{
	int n;
	while (scanf("%d", &n)!=EOF)
	{
		if (n & 1)
		{
			printf("Ehab\n");
		}
		else
		{
			printf("Mahmoud\n");
		}
	}
}