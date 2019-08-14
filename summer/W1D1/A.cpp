#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

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