/*
Centering and scaling functions.

Every function accepts a vector of length N and returns a transformed
vector with the same length.

Functions included:
- center
- scale
- standardize
- center_at
- scale_at
- rescale
- min_max
*/

// @laplace
// @brief Removes the sample mean from each observation.
// @param x Vector of length N.
// @return Vector of length N centered around zero.
// @example centered_x = transformation::center(x);
// @math
// x_i^{*} = x_i - \bar{x}
vector center(vector x){
    return x - mean(x);
}

// @laplace
// @brief Scales a variable relative to its sample standard deviation.
// @param x Vector of length N.
// @return Vector of length N scaled by the sample standard deviation.
// @example scaled_x = transformation::scale(x);
// @math
// x_i^{*} = \frac{x_i}{s_x}
vector scale(vector x){
    return x / sd(x);
}

// @laplace
// @brief Standardizes a variable by subtracting its mean and dividing by its standard deviation.
// @param x Vector of length N.
// @return Vector of length N with mean approximately zero and standard deviation one.
// @example standard_x = transformation::standardize(x);
// @math
// x_i^{*} = \frac{x_i - \bar{x}}{s_x}
vector standardize(vector x){
    return (x - mean(x)) / sd(x);
}

// @laplace
// @brief Centers observations around a specified reference value.
// @param x Vector of length N.
// @param c Real reference value used as the center.
// @return Vector of length N centered around c.
// @example centered_x = transformation::center_at(x, 10);
// @math
// x_i^{*} = x_i - c
vector center_at(vector x, real c){
    return x - c;
}

// @laplace
// @brief Scales observations using a specified reference scale.
// @param x Vector of length N.
// @param c Real reference scale.
// @return Vector of length N divided by the reference scale.
// @example scaled_x = transformation::scale_at(x, 10);
// @math
// x_i^{*} = \frac{x_i}{c}
vector scale_at(vector x, real c){
    return x / c;
}

// @laplace
// @brief Linearly maps a variable to an arbitrary interval [a, b].
// @param x Vector of length N.
// @param a Real lower bound of the target interval.
// @param b Real upper bound of the target interval.
// @return Vector of length N mapped from [min(x), max(x)] to [a, b].
// @example mapped_x = transformation::rescale(x, 0, 10);
// @math
// x_i^{*} = a + (b-a)\frac{x_i-\min(x)}{\max(x)-\min(x)}
vector rescale(vector x, real a, real b) {
    real x_min = min(x);
    real x_max = max(x);
    vector[size(x)] scaled;

    scaled = a + (b - a) * (x - x_min) / (x_max - x_min);

    return scaled;
}

// @laplace
// @brief Maps the minimum and maximum of a variable to 0 and 1 respectively.
// @param x Vector of length N.
// @return Vector of length N scaled to the interval [0, 1].
// @example mapped_x = transformation::min_max(x);
// @math
// x_i^{*} = \frac{x_i-\min(x)}{\max(x)-\min(x)}
vector min_max(vector x){
    return (x - min(x)) / (max(x) - min(x));
}


/*
Power transformations.

These functions accept a vector x of length N and return a transformed
vector with the same length.

Functions included:
- power_transform
- box_cox
- yeojohnson
*/

// @laplace
// @brief Applies a general power transformation to modify skewness or the relationship between a predictor and response.
// @param x Vector of length N. Values must be positive.
// @param lambda Real power controlling the transformation.
// @return Vector of length N containing the power-transformed values.
// @example power_x = transformation::power_transform(x, lambda);
// @math
// T(x_i;\lambda) =
// \begin{cases}
// x_i^\lambda, & \lambda \neq 0 \\
// \log(x_i), & \lambda = 0
// \end{cases}
vector power_transform(vector x, real lambda){
    int size_x = size(x);
    vector[size_x] y;

    for(i in 1:size_x){
        if(lambda == 0){
            y[i] = log(x[i]);
        }
        else{
            y[i] = pow(x[i], lambda);
        }
    }

    return y;
}

// @laplace
// @brief Applies the Box-Cox power transformation to positive continuous variables.
// @param x Vector of length N. All values must be strictly positive.
// @param lambda Real power controlling the transformation.
// @return Vector of length N containing the Box-Cox transformed values.
// @example boxcox_x = transformation::box_cox(x, lambda);
// @math
// T(x_i;\lambda) =
// \begin{cases}
// \frac{x_i^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
// \log(x_i), & \lambda = 0
// \end{cases}
vector box_cox(vector x, real lambda){
    int size_x = size(x);
    vector[size_x] y;

    for(i in 1:size_x){
        if(lambda == 0){
            y[i] = log(x[i]);
        }
        else{
            y[i] = (pow(x[i], lambda) - 1) / lambda;
        }
    }

    return y;
}

// @laplace
// @brief Applies the Yeo-Johnson power transformation to variables that may contain positive, zero, and negative values.
// @param x Vector of length N.
// @param lambda Real power controlling the transformation.
// @return Vector of length N containing the Yeo-Johnson transformed values.
// @example yeojohnson_x = transformation::yeojohnson(x, lambda);
// @math
// T(x_i;\lambda) =
// \begin{cases}
// \frac{(x_i+1)^\lambda-1}{\lambda}, & x_i \geq 0,\ \lambda \neq 0 \\
// \log(x_i+1), & x_i \geq 0,\ \lambda = 0 \\
// -\frac{(1-x_i)^{2-\lambda}-1}{2-\lambda}, & x_i < 0,\ \lambda \neq 2 \\
// -\log(1-x_i), & x_i < 0,\ \lambda = 2
// \end{cases}
vector yeojohnson(vector x, real lambda){
    int size_x = size(x);
    vector[size_x] y;

    for(i in 1:size_x){
        // For x >= 0
        if(x[i] >= 0){
            if(lambda == 0){
                y[i] = log(x[i] + 1);
            }
            else{
                y[i] = (pow(x[i] + 1, lambda) - 1) / lambda;
            }
        }
        // For x < 0
        if(x[i] < 0){
            if(lambda == 2){
                y[i] = -log(1 - x[i]);
            }
            else{
                y[i] = -((pow(1 - x[i], 2 - lambda) - 1) / (2 - lambda));
            }
        }
    }
    return y;
}


/*
Rank / empirical transformations.

These functions compare observations with each other. The original
magnitudes are replaced by information about the relative ordering
of observations.

Functions included:
- rank
- percent_rank
- uniformize
- normal_score
*/

// @laplace
// @brief Internal indicator function returning one when xj is smaller than xi and zero otherwise.
// @param xi Value being ranked.
// @param xj Value being compared with xi.
// @return Integer 1 if xi > xj and 0 otherwise.
// @math
// I(x_i,x_j) =
// \begin{cases}
// 1, & x_j < x_i \\
// 0, & x_j \geq x_i
// \end{cases}
int I(real xi, real xj) {
    int c;

    if (xj < xi) {
        c = 1;
    } else {
        c = 0;
    }

    return c;
}

// @laplace
// @brief Assigns each observation its rank among all observations.
// @param x Vector of length N.
// @return Vector containing the rank of each observation. Tied observations receive the same minimum rank.
// @example ranked_x = transformation::rank(x);
// @math
// r_i = 1 + \sum_{j=1}^{N} I(x_i,x_j)
vector rank(vector x) {
    int size_x = size(x);
    vector[size_x] ri;

    for (i in 1:size_x) {
        ri[i] = 1;

        for (j in 1:size_x) {
            ri[i] += I(x[i], x[j]);
        }
    }

    return ri;
}

// @laplace
// @brief Converts observation ranks to a relative position on the interval [0, 1].
// @param x Vector of length N.
// @return Vector containing the percent rank of each observation.
// @example percent_rank_x = transformation::percent_rank(x);
// @math
// p_i = \frac{r_i - 1}{N - 1}
vector percent_rank(vector x){
    int size_x = size(x);
    vector[size_x] ri = rank(x);
    vector[size_x] pr = (ri - 1) / (size_x - 1);

    return pr;
}

// @laplace
// @brief Maps empirical observations into the open interval (0, 1) using their ranks.
// @param x Vector of length N.
// @return Vector of empirical probabilities strictly between 0 and 1.
// @example uniformed_rank_x = transformation::uniformize(x);
// @math
// u_i = \frac{r_i - \frac{1}{2}}{N}
vector uniformize(vector x){
    int size_x = size(x);
    vector[size_x] ri = rank(x);
    vector[size_x] ui = (ri - 0.5) / size_x;

    return ui;
}

// @laplace
// @brief Maps empirical ranks onto an approximately standard-normal scale using the inverse standard-normal CDF.
// @param x Vector of length N.
// @return Vector of normal-score transformed observations.
// @example normal_score_x = transformation::normal_score(x);
// @math
// z_i = \Phi^{-1}\left(\frac{r_i-\frac{1}{2}}{N}\right)
vector normal_score(vector x){
    int size_x = size(x);
    vector[size_x] ui = uniformize(x);
    vector[size_x] zi = inv_Phi(ui);

    return zi;
}


/*
Clipping function.

Takes a vector of length N and restricts values below a lower
bound or above an upper bound to those respective bounds.
*/

// @laplace
// @brief Restricts extreme values to a specified lower and upper bound.
// @param x Vector of length N.
// @param low Real lower bound.
// @param up Real upper bound.
// @return Vector of length N with all values restricted to [low, up].
// @example clipped_x = transformation::clip(x, 0, 100);
// @math
// T(x_i) =
// \begin{cases}
// low, & x_i < low \\
// x_i, & low \leq x_i \leq up \\
// up, & x_i > up
// \end{cases}
vector clip(vector x, real low, real up){
    int size_x = size(x);
    vector[size_x] clipped;

    for(i in 1:size_x){
        if(x[i] > up){
            clipped[i] = up;
        }
        else if (x[i] < low) {
            clipped[i] = low;
        }
        else{
            clipped[i] = x[i];
        }
    }

    return clipped;
}