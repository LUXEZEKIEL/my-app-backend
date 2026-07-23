import chromadb

try:
    # 1. 连接本地 Docker 里的 ChromaDB
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000)

    # 2. 获取你刚才创建的心理学数据集
    collection = chroma_client.get_collection(name="psychology_rules")

    # 3. 打印目前库里一共有多少条数据
    count = collection.count()
    print("\n" + "="*50)
    print(f"📊 检查成功！当前 ChromaDB 数据集中共有数据量: {count} 条")
    print("="*50 + "\n")

    # 4. 把所有存进去的文档内容全部拉出来展示
    results = collection.get()
    for i, (doc_id, doc_text) in enumerate(zip(results['ids'], results['documents'])):
        print(f"📌 [文档 {i+1} ID]: {doc_id}")
        print(f"📝 [文档内容]:\n{doc_text.strip()}")
        print("-" * 50)

except Exception as e:
    print("\n❌ 检查失败，错误信息如下：")
    print(e)