#include<cstdio>
#include<cstring>
#include<string>
#include<iostream>
using namespace std;
struct node {
	node *next[26];
	bool isStr;
    string tran;
};
node* T;
void insert(node* root, string s, string tr)
{
	node* p = root;
	for (int i = 0; i < s.size(); i++)
	{
		if (p->next[s[i] - 'a'] == NULL)
		{
			node* temp = new node;
			for (int j = 0; j < 26; j++)
			{
				temp->next[j] = NULL;
			}
			temp->isStr = false;
			p->next[s[i] - 'a'] = temp;
			p = p->next[s[i] - 'a'];
		}
		else
		{
			p = p->next[s[i] - 'a'];
		}
	}
	p->isStr = true;
    p->tran=tr;
}

string find(node* root, string s)
{
	node *p = root;
	for (int i = 0; i < s.size(); i++)
	{
		if (p->next[s[i] - 'a'] == NULL)
		{
			return "eh";
		}
		else
		{
			p = p->next[s[i] - 'a'];
		}
	}
	if (p->isStr)
		return p->tran;
	else
		return "eh";
}
int main()
{
	//freopen("in", "r", stdin);
	T = new node;
	for (int i = 0; i < 26; i++)
		T->next[i] = NULL;
	int n, m;
	scanf("%d%d", &n, &m);
    char c;
    int f=0;
    int f1=1;
    string s,tr;
    while(c=getchar())
    {
        if(f&&c=='\n') break;
        else f=0;
        if(c!=' '&&c!='\n')
        {
            if(f1==2) s+=c;
            else tr+=c;
        }
        if(c==' ') f1=2;
        if(c=='\n')
        {
            insert(T, s, tr);
            s.clear();
            tr.clear();
            f1=1;
            f=1;
        }
    }
    while(cin>>s)
    {
        cout<<find(T, s)<<endl;
    }
	return 0;
}