#include <cstdio>
struct Node{ 
   int key;
   int priority;
   Node* left;
   Node* right;
};
Node* root;
void rotate_left(Node* &node)
{
	Node* x = node->right;
	node->right = x->left;
	x->left =node;
	node = x;
}
void rotate_right(Node* &node)
{
	Node* x = node->left;
	node->left = x->right;
	x->right = node;
	node = x;
}

void treap_insert(Node* &root, int key, int priority)
{
	if (root == NULL)
	{
		root = (Node*)new Node;
		root->left = NULL;
		root->right = NULL;
		root->priority = priority;
		root->key = key;
	}
	else if (key <root->key)
	{
		treap_insert(root->left, key, priority);
		if (root->left->priority < root->priority)
			rotate_right(root);
	}
	else
	{
		treap_insert(root->right, key, priority);
		if (root->right->priority < root->priority)
			rotate_left(root);
	}
}

int find(Node* &root, int key)
{
    if(root->key==key) return 1;
    if(root->left==NULL&&key<root->key) return 0;
    if(root->right==NULL&&key>root->key) return 0;
    else
    {
        if(root->key<key) return find(root->right, key);
        else return find(root->left, key);
    }
}
int max(int a,int b)
{
    if(a>b) return a;
    else return b;
}
int pre(Node* &root, int key)
{
    if(root==NULL) return -1000000001;
    if(root->key>key) return pre(root->left, key);
    return max(root->key, pre(root->right, key));
}
int rand()
{
    static int seed = 703;
    return seed = int(seed*48271LL%(~0u>>1));
}
int main()
{
    int n,x;
    char op;
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    {
        getchar();
        scanf("%c%d",&op,&x);
        if(op=='I')
        {
            treap_insert(root, x, rand());
        }else
        {
            if(find(root, x)) printf("%d\n",x);
            else printf("%d\n",pre(root, x));
        }
    }
    return 0;
}