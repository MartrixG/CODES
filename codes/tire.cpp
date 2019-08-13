#include<cstdio>
#include<cstring>
#include<string>
#include<iostream>
using namespace std;
struct node {
	node *next[26];
	bool isStr;
};
node* T;
void insert(node* root, string s)
{
	node* p = root;
	for (int i = 0; i < s.size(); i++)
	{
		if (p->next[s[i] - 'a'] == NULL)
		{
			node* temp = new node;
			for (int i = 0; i < 26; i++)
			{
				temp->next[i] = NULL;
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

}

bool find(node* root, string s)
{
	node *p = root;
	for (int i = 0; i < s.size(); i++)
	{
		if (p->next[s[i] - 'a'] == NULL)
		{
			return false;
		}
		else
		{
			p = p->next[s[i] - 'a'];
		}
	}
	if (p->isStr)
		return true;
	else
		return false;
}
int main()
{
	freopen("in", "r", stdin);
	T = new node;
	for (int i = 0; i < 26; i++)
		T->next[i] = NULL;
	int n, m;
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)
	{
		string s;
		cin >> s;
		insert(T, s);
	}
	for (int i = 1; i <= m; i++)
	{
		string s;
		cin >> s;
		printf("%d\n", find(T, s));
	}
}