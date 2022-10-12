import streamlit as st

"""
code displayed under every function that returns something

example
def df_head():
    st.code('''df.head''', language='python)
"""

#head
def df_head():
    st.code('''df.head()''', language='python')

#shape
def df_shape():
    st.code('''df.shape''', language='python')

#plot target
def plot_target():
    st.code("""
                fig, ax = plt.subplots()
                sns.countplot(data = train_data, x="target")
                plt.show()
    """,
    language='python')


def heatmap_code():
    st.code('''
    train_data.corr()["target"].sort_values(by="target", ascending=False)
    ''', language='python')

def heatmap_sns():
    st.code('''
    sns.heatmap(train.corr(), annot=True)
    ''', language='python')