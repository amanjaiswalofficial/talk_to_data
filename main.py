from load_store import create_qa_chain, generate_response

qa_chain = create_qa_chain()

while True:
    user_query = input('Input query: ')
    response = generate_response(query=user_query, 
                                 qa_chain=qa_chain)
    print(response)
