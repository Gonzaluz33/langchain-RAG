from flask import Blueprint, request, jsonify
from agent import agent, keyword_chain

routes_bp = Blueprint('routes', __name__)

@routes_bp.route("/query", methods=["POST"])
def get_response():
    data = request.get_json()
    query = data.get("query", "")
    print(f"Consulta recibida: {query}")
    keywords = keyword_chain.run(query=query)
    print("Keywords generadas:", keywords)
    agente_input = f"La consulta del usuario es: '{query}'. Las keywords son: {keywords}"
    final_response = agent.run(agente_input)
    return jsonify({"query": query, "response": final_response})
