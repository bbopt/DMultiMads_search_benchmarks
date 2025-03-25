/**
  \file   MOP3.cpp
  \brief  Library example for nomad (DMulti-MADS algorithm)
  \author Ludovic Salomon
  \date   2024
  */
#include "Cache/CacheBase.hpp"
#include "Nomad/nomad.hpp"
#include "Type/DMultiMadsSearchStrategyType.hpp"
#include <cmath>

class MOP3 : public NOMAD::Evaluator
{
  public:
    explicit MOP3(const std::shared_ptr<NOMAD::EvalParameters> &evalParams)
        : NOMAD::Evaluator(evalParams, NOMAD::EvalType::BB)
    {
    }

    ~MOP3() override = default;

    bool eval_x(NOMAD::EvalPoint &x, const NOMAD::Double &hMax, bool &countEval) const override
    {
        bool eval_ok = false;

        NOMAD::Double f1 = 1e20, f2 = 1e20;

        try
        {
            // Parameters
            const double A1 = 0.5 * sin(1.0) - 2 * cos(1.0) + sin(2.0) - 1.5 * cos(2.0);
            const double A2 = 1.5 * sin(1.0) - cos(1.0) + 2 * sin(2.0) - 0.5 * cos(2.0);

            // Variables
            double x1 = x[0].todouble();
            double x2 = x[1].todouble();

            // Objectives
            // Additional variables
            const double B1 = 0.5 * sin(x1) - 2 * cos(x1) + sin(x2) - 1.5 * cos(x2);
            const double B2 = 1.5 * sin(x1) - cos(x1) + 2 * sin(x2) - 0.5 * cos(x2);

            f1 = -1 - (A1 - B1) * (A1 - B1) - (A2 - B2) * (A2 - B2);
            f2 = -(x1 + 3) * (x1 + 3) - (x2 + 1) * (x2 + 1);

            // Maximization
            std::string bbo = (-f1).tostring() + " " + (-f2).tostring();

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
};

bool run_pb()
{
    NOMAD::MainStep TheMainStep;

    try
    {
        // Parameters creation
        auto params = std::make_shared<NOMAD::AllParameters>();

        const double PI = 3.141592653589793238463;

        // Dimensions of the blackbox, inputs and outputs
        const size_t n = 2;
        params->setAttributeValue("DIMENSION", n);

        NOMAD::ArrayOfDouble lb(n, -PI);
        params->setAttributeValue("LOWER_BOUND", lb);

        NOMAD::ArrayOfDouble ub(n, PI);
        params->setAttributeValue("UPPER_BOUND", ub);

        // Outputs, constraints and objectives
        NOMAD::BBOutputTypeList bbOutputTypes;
        for (size_t i = 0; i < 2; ++i)
        {
            bbOutputTypes.emplace_back(NOMAD::BBOutputType::OBJ);
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
        std::string history_name = "MOP3_dmultimads.txt";
        params->setAttributeValue("HISTORY_FILE", history_name); // Save history. To uncomment if you want it.

        // Validate
        params->checkAndComply();

        // Run the solver
        TheMainStep.setAllParameters(params);
        auto ev = std::make_unique<MOP3>(params->getEvalParams());
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
    run_pb();
    return 0;
}
