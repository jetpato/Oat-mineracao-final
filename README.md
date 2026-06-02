# 🛒 Sistema Completo — Recomendador de Supermercado

**Grupo:** Alisson Silva, Ana Luiza Lima, Julio Cesar Brito, João Pedro Ceo, Mateus Dantas e Levi Ramos

Modelo: **Algoritmo Apriori (Regras de Associação)**

---

## Arquivos do projeto

| Arquivo | O que é |
|---|---|
| `parte1_salvar_modelo.ipynb` | Notebook que treina, salva (`.pkl`) e verifica o modelo |
| `modelo_supermercado.pkl` | Modelo treinado e salvo (já gerado, pronto para usar) |
| `app.py` | Interface web em Streamlit |

---

## As três camadas do sistema

1. **Modelo treinado e salvo** → `modelo_supermercado.pkl` (gerado pela Parte 1 com joblib)
2. **Back-end** → o código dentro do `app.py` que carrega o modelo, aplica o pré-processamento e devolve a previsão
3. **Interface** → a tela do Streamlit onde o usuário monta o carrinho e vê a recomendação

---

## Sobre o scaler (importante para o professor)

O enunciado pede para salvar o `StandardScaler`/`MinMaxScaler` **quando aplicável**.

No nosso projeto **não há scaler**: os dados já são **binários (0 = não comprou / 1 = comprou)**, então
nunca foi usada normalização. Por isso salvamos apenas o modelo (as regras) e os metadados.
No `.pkl` o campo `scaler` está como `None` para deixar isso explícito.

---

## Como executar (passo a passo)

### Parte 1 — Gerar o modelo (já feito)
O arquivo `modelo_supermercado.pkl` já está pronto. Se quiser gerar de novo:
1. Abra `parte1_salvar_modelo.ipynb` no Google Colab
2. Execute todas as células (Ctrl + F9)
3. A última célula baixa o `.pkl`

### Parte 2 — Rodar a interface
No terminal, dentro da pasta onde estão `app.py` e `modelo_supermercado.pkl`:

```bash
pip install streamlit mlxtend joblib pandas
streamlit run app.py
```

O navegador abre sozinho em `http://localhost:8501`.

---

## Roteiro para a apresentação ao vivo

Mostrar **dois exemplos de entrada diferentes**:

| Exemplo | Entrada (carrinho) | Resultado esperado |
|---|---|---|
| 1 | Arroz | Recomenda **Feijão** com 67% de confiança |
| 2 | Pão | Recomenda **Leite** com 100% de confiança |

(Opcional) Mostrar um terceiro caso: **Café** → o sistema responde que não há recomendação confiável,
demonstrando que o modelo não inventa padrões para produtos raros.
