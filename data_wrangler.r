library(dplyr)
library(ggplot2)
library(ggcharts)
library(reshape2)
library(purrr)
library(lazyeval)
library(yaml)
library(data.table)
library(formattable)
library(pROC)

df <- read.csv('data/dataset.csv')

PARAMS = list(
    fee = 0.003,
    min_cb = 0.025,
    relevant_columns = c('billing_country_code', 'shipping_country_code', 'gateway'),
    cols = list(
      customGreen0 = "#DeF7E9",
      customGreen = "#71CA97",
      customRed = "#ff7f7f")
  )

kpi_calculator <- function(df){
  output <- df %>%
    summarise(total_orders = n(),
              total_approved_volume = round(sum(total_spent[status != 'declined'])),
              approval_rate = percent(sum(status != 'declined')/n()),
              decline_rate = percent(1-approval_rate),
              cb_rate = percent(sum(status == 'chargeback')/sum(status != 'declined')),
              cb_paid = round(sum(total_spent[status == 'chargeback'])),
              ) %>%
    arrange(desc(cb_rate))
  return(output)
}


kpi_calculator_multi <- function(col_grp){
  output <- df %>%
    group_by(across(col_grp)) %>%
    summarise(total_orders = n(),
              total_approved_volume = round(sum(total_spent[status != 'declined'])),
              approval_rate = percent(sum(status != 'declined')/n()),
              decline_rate = percent(1-approval_rate),
              cb_rate = percent(sum(status == 'chargeback')/sum(status != 'declined')),
              cb_paid = round(sum(total_spent[status == 'chargeback'])),
              ) %>%
    arrange(desc(cb_rate)) %>% ungroup()
  return(output)
}


fixed_calcultor_output <- function(x){
  data <- kpi_calculator_multi(df, x)
  variable <- names(data)[1]
  names(data)[1] <- 'variable'
  data$var_type <- variable
  return(data)
}

plot_hist <- function(df, x_col, fill_col){
  output<- df %>%
    ggplot(aes_string(x=x_col, fill=fill_col)) +
    geom_density(alpha = 0.5) +
    theme_hermit() +
    labs(title = 'Total $ Spent - Distribution',
         subtitle = paste0('by ', fill_col),
         fill = '',
         color = '') +
    theme(legend.position = 'top')
  return(output)
}

level_counter <- function(col){
  loc = which(names(df) == col)
  return(length(unique(df[,loc])))
}

d_type <- function(col){
  loc = which(names(df) == col)
  return(class(unique(df[,loc])))
}

get_grouping_set <- function(x){
  return(split(x, rep(1:ncol(x), each = nrow(x))))
}

model_performence_comp <- function(df, string){
  col_name = ifelse(string == 'riski', 'Riskified', 'Merchant')
  kpis <- df %>%
    select(order_count, grep(string,names(performence_kpi))) %>%
    reshape2:: melt() %>% 
    mutate(variable = gsub(paste0('_',string), '', variable),
           value = ifelse(value < 1, 
                          paste0(round(value, 4)*100, '%'),
                          round(value, 0))) %>%
    rename(!!sym(col_name) := value)
  return(kpis)
}


