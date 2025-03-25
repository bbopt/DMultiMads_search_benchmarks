/**
  \file   ZDT3.cpp
  \brief  Library example for nomad (DMulti-MADS algorithm)
  \author Ludovic Salomon
  \date   2024
  */
#include "Cache/CacheBase.hpp"
#include "Nomad/nomad.hpp"
#include "Type/DMultiMadsSearchStrategyType.hpp"
#include <cmath>

class ZDT3 : public NOMAD::Evaluator
{
  public:
    explicit ZDT3(const std::shared_ptr<NOMAD::EvalParameters> &evalParams, const size_t n, const size_t ctype)
        : NOMAD::Evaluator(evalParams, NOMAD::EvalType::BB), _n(n), _ctype(ctype)
    {
    }

    ~ZDT3() override = default;

    bool eval_x(NOMAD::EvalPoint &x, const NOMAD::Double &hMax, bool &countEval) const override
    {
        bool eval_ok = false;

        NOMAD::Double f1 = 1e20, f2 = 1e20;

        try
        {
            const size_t n = _n;

            // Parameters
            const double PI = 3.141592653589793238463;

            // Variables
            std::vector<double> xi(n, 0);
            for (int i = 0; i < _n; ++i)
            {
                xi[i] = x[i].todouble();
            }

            // Objectives
            f1 = xi[0];

            double g = 1.0;
            for (int i = 1; i < n; ++i)
            {
                g += (9.0 / (n - 1)) * xi[i];
            }
            const double h = 1.0 - sqrt(f1.todouble() / g) - (f1.todouble() / g) * sin(10 * PI * f1.todouble());
            f2 = g * h;

            std::string bbo = f1.tostring() + " " + f2.tostring() + " ";

            // Constraints
            if (_ctype == 1)
            {
                int l = _n - 2;
                for (int j = 0; j < l; ++j)
                {
                    NOMAD::Double c = (3 - 2 * xi[j + 1]) * xi[j + 1] - xi[j] - 2 * xi[j + 2] + 1;
                    bbo += c.tostring() + " ";
                }
            }
            else if (_ctype == 2)
            {
                int l = _n - 2;
                for (int j = 0; j < l; ++j)
                {
                    NOMAD::Double c = (3 - 2 * xi[j + 1]) * xi[j + 1] - xi[j] - 2 * xi[j + 2] + 2.5;
                    bbo += c.tostring() + " ";
                }
            }
            else if (_ctype == 3)
            {
                int l = _n - 1;
                for (int j = 0; j < l; ++j)
                {
                    NOMAD::Double c =
                        xi[j] * xi[j] + xi[j + 1] * xi[j + 1] + xi[j] * xi[j + 1] - 2 * xi[j] - 2 * xi[j + 1] + 1;
                    bbo += c.tostring() + " ";
                }
            }
            else if (_ctype == 4)
            {
                int l = _n - 1;
                for (int j = 0; j < l; ++j)
                {
                    NOMAD::Double c = xi[j] * xi[j] + xi[j + 1] * xi[j + 1] + xi[j] * xi[j + 1] - 1;
                    bbo += c.tostring() + " ";
                }
            }
            else if (_ctype == 5)
            {
                int l = _n - 2;
                for (int j = 0; j < l; ++j)
                {
                    NOMAD::Double c = (3 - 0.5 * xi[j + 1]) * xi[j + 1] - xi[j] - 2 * xi[j + 2] + 1;
                    bbo += c.tostring() + " ";
                }
            }
            else if (_ctype == 6)
            {
                int l = _n - 2;
                NOMAD::Double c = 0.0;
                for (int i = 0; i < l; ++i)
                {
                    c += ((3 - 0.5 * xi[i + 1]) * xi[i + 1] - xi[i] - 2 * xi[i + 2] + 1);
                }
                bbo += c.tostring();
            }

            x.setBBO(bbo);

            eval_ok = true;
        }
        catch (std::exception &e)
        {
            std::string err("Exception: ");
            err += e.what();
            throw std::logic_error(err);
        }
        countEval = true;
        return eval_ok;
    }

  private:
    size_t _n;
    size_t _ctype;
};

bool run_pb(const size_t n, const size_t ctype)
{
    NOMAD::MainStep TheMainStep;

    try
    {
        // Parameters creation
        auto params = std::make_shared<NOMAD::AllParameters>();

        // Dimensions of the blackbox, inputs and outputs
        params->setAttributeValue("DIMENSION", n);

        NOMAD::ArrayOfDouble lb(n, 0);
        params->setAttributeValue("LOWER_BOUND", lb);

        NOMAD::ArrayOfDouble ub(n, 1);
        params->setAttributeValue("UPPER_BOUND", ub);

        // Outputs, constraints and objectives
        NOMAD::BBOutputTypeList bbOutputTypes;
        for (size_t i = 0; i < 2; ++i)
        {
            bbOutputTypes.emplace_back(NOMAD::BBOutputType::OBJ);
        }
        if (ctype == 1 || ctype == 2 || ctype == 5)
        {
            int l = n - 2;
            for (int i = 0; i < l; ++i)
            {
                bbOutputTypes.emplace_back(NOMAD::BBOutputType::PB);
            }
        }
        else if (ctype == 3 || ctype == 4)
        {
            int l = n - 1;
            for (int i = 0; i < l; ++i)
            {
                bbOutputTypes.emplace_back(NOMAD::BBOutputType::PB);
            }
        }
        else if (ctype == 6)
        {
            bbOutputTypes.emplace_back(NOMAD::BBOutputType::PB);
        }
        params->setAttributeValue("BB_OUTPUT_TYPE", bbOutputTypes);

        // A line initialization is practically more efficient than giving a single point for
        // multiobjective optimization.
        // The interesting reader can report to the following reference for more information.
        // Direct Multisearch for multiobjective optimization
        // by A.L. Custodio, J.F.A. Madeira, A.I.F. Vaz and L.N. Vicente, 2011.
        NOMAD::ArrayOfPoint x0s;
        for (size_t j = 0; j < n; ++j)
        {
            NOMAD::Point x0(n, 0);
            for (size_t i = 0; i < n; ++i)
            {
                x0[i] = lb[i] + (double)j * (ub[i] - lb[i]) / (n - 1);
            }
            x0s.push_back(x0);
        }
        // Starting point
        params->setAttributeValue("X0", x0s);

        // Algorithm parameters
        // 1- Terminate after this number of maximum blackbox evaluations.
        params->setAttributeValue("MAX_BB_EVAL", 30000);

        // 2- Use n+1 directions
        params->setAttributeValue("DIRECTION_TYPE", NOMAD::DirectionType::ORTHO_NP1_NEG);

        // 3- For multiobjective optimization, these parameters are required.
        params->setAttributeValue("DMULTIMADS_OPTIMIZATION", true);
        // For multiobjective optimization, sort cannot use the default quad model info.
        params->setAttributeValue("EVAL_QUEUE_SORT", NOMAD::EvalSortType::DIR_LAST_SUCCESS);

        // Special search
        params->setAttributeValue("NM_SEARCH", false); // Deactivate NM search
        // params->setAttributeValue("DMULTIMADS_NM_STRATEGY", NOMAD::DMultiMadsNMSearchType::DOM);
        params->setAttributeValue("DMULTIMADS_NM_STRATEGY", NOMAD::DMultiMadsNMSearchType::MULTI);
        params->setAttributeValue("QUAD_MODEL_SEARCH", false); // Deactivate Quad model search
        params->setAttributeValue("DMULTIMADS_QUAD_MODEL_STRATEGY", NOMAD::DMultiMadsQuadSearchType::DMS);
        // params->setAttributeValue("DMULTIMADS_QUAD_MODEL_STRATEGY", NOMAD::DMultiMadsQuadSearchType::DOM);
        // params->setAttributeValue("DMULTIMADS_QUAD_MODEL_STRATEGY", NOMAD::DMultiMadsQuadSearchType::MULTI);
        params->setAttributeValue("QUAD_MODEL_SEARCH_SIMPLE_MADS", false);
        params->setAttributeValue("DMULTIMADS_QMS_PRIOR_COMBINE_OBJ", true);
        // params->setAttributeValue("DMULTIMADS_EXPANSIONINT_LINESEARCH", true);
        // params->setAttributeValue("DMULTIMADS_MIDDLEPOINT_SEARCH, true);

        // Advanced attributes for DMultiMads
        params->setAttributeValue("DMULTIMADS_SELECT_INCUMBENT_THRESHOLD", 1);

        // 4- Other useful parameters
        params->setAttributeValue("DISPLAY_DEGREE", 2);
        params->setAttributeValue("SOLUTION_FILE", std::string("sol.txt")); // Save the Pareto front approximation
        std::string history_name = "ZDT3_";
        history_name += std::to_string(ctype) + "_dmultimads.txt";
        params->setAttributeValue("HISTORY_FILE", history_name); // Save history. To uncomment if you want it.

        // Validate
        params->checkAndComply();

        // Run the solver
        TheMainStep.setAllParameters(params);
        auto ev = std::make_unique<ZDT3>(params->getEvalParams(), n, ctype);
        TheMainStep.addEvaluator(std::move(ev));

        TheMainStep.start();
        TheMainStep.run();
        TheMainStep.end();
        NOMAD::MainStep::resetComponentsBetweenOptimization();
        return true;
    }

    catch (std::exception &e)
    {
        std::cerr << "\nNOMAD has been interrupted (" << e.what() << ")\n\n";
    }
    return false;
}

// Main function
int main(int argc, char **argv)
{
    const size_t n = 30;
    //std::vector<size_t> ctype{1, 3, 4};
    std::vector<size_t> ctype{0};
    for (const auto celt : ctype)
    {
        run_pb(n, celt);
    }
    return 0;
}
