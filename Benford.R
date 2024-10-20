# Instalar o pacote BenfordTests, se ainda não estiver instalado
install.packages("BenfordTests")
install.packages('benford.analysis')
install.packages ("dplyr")
install.packages("readxl")

# Carregar as bibliotecas necessárias
library(benford.analysis)
library(BenfordTests)
library(dplyr)
library(readxl)

# Carregar o arquivo XLSX
dados <- read_excel("202401_NFe_NotaFiscalItem.xlsx")

colnames(dados) <- make.names(colnames(dados))

dados_filtrados <- dados %>%
  filter(`NCM.SH..TIPO.DE.PRODUTO.` %in% c("Gasóleo (óleo diesel)", "Outras gasolinas, exceto para aviação"))


# Realizar a análise da Lei de Benford agrupada por CNPJ DESTINATÁRIO e DESCRIÇÃO DO PRODUTO/SERVIÇO
resultado <- benford(dados_filtrados$QUANTIDADE,1)
plot(resultado)


# Agrupar os dados por "RAZÃO.SOCIAL.EMITENTE" e aplicar a análise de Benford em cada grupo
resultados <- dados_filtrados %>%
  group_by(RAZÃO.SOCIAL.EMITENTE) %>%
  summarise(resultado_benford = list(benford(QUANTIDADE, 2)))

# Plotar os resultados para cada grupo
for (i in seq_along(resultados$resultado_benford)) {
  plot(resultados$resultado_benford[[i]], main = paste("Benford para", resultados$RAZÃO.SOCIAL.EMITENTE[i]))
}


