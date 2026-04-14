"""
Sistema de Recomendação para E-commerce
Técnica: Filtragem Colaborativa (Collaborative Filtering)
Baseado na aula de Deep Learning - Munif Gebara
"""

import numpy as np
from collections import defaultdict


historico = {
    "usuario_1": ["tenis_nike", "meia", "shorts"],
    "usuario_2": ["tenis_nike", "camiseta", "bone"],
    "usuario_3": ["meia", "shorts", "camiseta"],
    "usuario_4": ["tenis_adidas", "shorts", "bone"],
    "usuario_5": ["tenis_nike", "meia", "bone"],
}

todos_produtos = list({p for compras in historico.values() for p in compras})
todos_usuarios = list(historico.keys())



def construir_matriz(historico, usuarios, produtos):
    """Cria matriz binária usuário x produto."""
    idx_usuario = {u: i for i, u in enumerate(usuarios)}
    idx_produto  = {p: i for i, p in enumerate(produtos)}

    matriz = np.zeros((len(usuarios), len(produtos)), dtype=float)

    for usuario, compras in historico.items():
        for produto in compras:
            i = idx_usuario[usuario]
            j = idx_produto[produto]
            matriz[i][j] = 1.0

    return matriz, idx_usuario, idx_produto

def similaridade_cosseno(vetor_a, vetor_b):
    """
    Calcula o quanto dois usuários têm gostos parecidos.
    Resultado: 0.0 (nada parecido) → 1.0 (idênticos)
    """
    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return float(np.dot(vetor_a, vetor_b) / (norma_a * norma_b))


def calcular_similaridades(matriz, idx_usuario_alvo):
    """Retorna a similaridade do usuário alvo com todos os outros."""
    similares = {}
    vetor_alvo = matriz[idx_usuario_alvo]

    for i, _ in enumerate(todos_usuarios):
        if i == idx_usuario_alvo:
            continue
        sim = similaridade_cosseno(vetor_alvo, matriz[i])
        similares[todos_usuarios[i]] = round(sim, 4)

    return dict(sorted(similares.items(), key=lambda x: x[1], reverse=True))


def recomendar(usuario_alvo, historico, k=3, top_n=3):
    """
    Recomenda produtos usando Filtragem Colaborativa.

    Parâmetros:
        usuario_alvo : str  — usuário que vai receber as recomendações
        historico    : dict — histórico de compras de todos os usuários
        k            : int  — quantidade de vizinhos mais próximos
        top_n        : int  — quantidade de produtos a recomendar

    Retorno:
        lista de tuplas (produto, score)
    """

    matriz, idx_usuario, idx_produto = construir_matriz(
        historico, todos_usuarios, todos_produtos
    )

    if usuario_alvo not in idx_usuario:
        print(f"Usuário '{usuario_alvo}' não encontrado.")
        return []

    idx_alvo = idx_usuario[usuario_alvo]

    # Produtos que o usuário já comprou (não recomendar de novo)
    ja_comprou = set(historico[usuario_alvo])

    # Similaridade com todos os outros usuários
    similares = calcular_similaridades(matriz, idx_alvo)

    # Pega os K vizinhos mais próximos
    vizinhos = list(similares.items())[:k]

    # Score ponderado por similaridade
    score_produto = defaultdict(float)

    for vizinho, sim in vizinhos:
        for produto in historico[vizinho]:
            if produto not in ja_comprou:
                score_produto[produto] += sim

    # Ordena por score e retorna top_n
    recomendacoes = sorted(score_produto.items(), key=lambda x: x[1], reverse=True)
    return recomendacoes[:top_n]


if __name__ == "__main__":

    usuario_teste = "usuario_1"

    print("=" * 45)
    print(f"  SISTEMA DE RECOMENDAÇÃO - E-COMMERCE")
    print("=" * 45)
    print(f"\nUsuário: {usuario_teste}")
    print(f"Já comprou: {historico[usuario_teste]}\n")

    # Similaridade com outros usuários
    matriz, idx_u, idx_p = construir_matriz(historico, todos_usuarios, todos_produtos)
    similares = calcular_similaridades(matriz, idx_u[usuario_teste])

    print("Usuários mais similares:")
    for u, sim in similares.items():
        barra = "█" * int(sim * 20)
        print(f"  {u:12s} | {barra:<20} {sim:.2f}")

    print("\nProdutos recomendados:")
    recomendacoes = recomendar(usuario_teste, historico, k=3, top_n=3)

    if recomendacoes:
        for rank, (produto, score) in enumerate(recomendacoes, start=1):
            print(f"  {rank}. {produto:20s} score: {score:.2f}")
    else:
        print("  Nenhuma recomendação disponível.")

    print("\n" + "=" * 45)
    print("Técnica: Filtragem Colaborativa (user-based)")
    print("Métrica: Similaridade do Cosseno")
    print("=" * 45)