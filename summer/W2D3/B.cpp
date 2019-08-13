#include<cstdio>
#include<cstring>
#include<iostream>
#include<cstdio>
#include<string>
using namespace std;
int f[400010];
int main()
{
    char m[400010];
    int n;
    scanf("%d",&n);
    while(scanf("%s",m)!=EOF)
    {
        memset(f,0,sizeof(f));
        int lm=strlen(m);
	    f[0] = -1;
	    int i = 0;
	    int j = -1;
	    while (i < lm)
	    {
		    while (j != -1 && m[i] != m[j])
		    {
			    j = f[j];
		    }
		    f[++i] = ++j;
	    }
        j=lm;
        int ans[400010];
        int tot=0;
        for(int i=0;i<lm;i++)
           // printf("%d ",f[i]);
        while(j!=0)
        {
            ans[++tot]=f[j];
            j=f[j];
        }
        for(int i=tot-1;i>=1;i--)
        {
            printf("%d ",ans[i]);
        }
        printf("%d\n",lm);
    }
    return 0;
}