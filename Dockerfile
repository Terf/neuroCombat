FROM python:3.10
ARG R_VERSION_MAJOR=4
ARG R_VERSION_MINOR=1
ARG R_VERSION_PATCH=2
ARG CONFIGURE_OPTIONS="--with-cairo --with-jpeglib --enable-R-shlib --with-blas --with-lapack"
RUN apt-get update && \
    apt-get install -y \
    gfortran \
    git \
    g++ \
    libreadline-dev \
    libxt-dev \
    libcairo2-dev \   
    libssl-dev \ 
    libxml2-dev \
    libudunits2-dev \
    libgdal-dev \
    libbz2-dev \
    libzstd-dev \
    liblzma-dev \
    libpcre2-dev \
    curl && \
    curl -LO https://cran.rstudio.com/src/base/R-${R_VERSION_MAJOR}/R-${R_VERSION_MAJOR}.${R_VERSION_MINOR}.${R_VERSION_PATCH}.tar.gz && \
    tar zxvf R-${R_VERSION_MAJOR}.${R_VERSION_MINOR}.${R_VERSION_PATCH}.tar.gz && \
    rm R-${R_VERSION_MAJOR}.${R_VERSION_MINOR}.${R_VERSION_PATCH}.tar.gz && \
    cd /R-${R_VERSION_MAJOR}.${R_VERSION_MINOR}.${R_VERSION_PATCH} && \
    ./configure ${CONFIGURE_OPTIONS} && \ 
    make && \
    make install && \
    echo 'options(repos = c(CRAN = "https://cran.rstudio.com/"), download.file.method = "libcurl")' >> /usr/local/lib/R/etc/Rprofile.site && \
    Rscript -e 'install.packages("devtools"); devtools::install_github("jfortin1/neuroCombat_Rpackage")' && \
    pip3 install rpy2 pandas numpy patsy
ENV LD_LIBRARY_PATH=/usr/local/lib/R/library/methods/libs:/usr/local/lib/R/lib:${LD_LIBRARY_PATH}
COPY ./dCombatR /opt/dCombatR
ENTRYPOINT [ "python3" ]
