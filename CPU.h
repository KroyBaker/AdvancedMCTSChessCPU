#ifndef CPU_H
#define CPU_H

#include <stddef.h>

#include "Board.h"

typedef struct {
    int pawn_mg;
    int knight_mg;
    int bishop_mg;
    int rook_mg;
    int queen_mg;
    int anteater_mg;
    int pawn_eg;
    int knight_eg;
    int bishop_eg;
    int rook_eg;
    int queen_eg;
    int anteater_eg;
    int center_mg;
    int center_eg;
    int pawn_advance_mg;
    int pawn_advance_eg;
    int mobility_mg;
    int mobility_eg;
    int pawn_structure_mg;
    int pawn_structure_eg;
    int passed_pawn_mg;
    int passed_pawn_eg;
    int king_safety_mg;
    int king_activity_eg;
    int threat_mg;
    int threat_eg;
    int rook_activity_mg;
    int rook_activity_eg;
    int minor_activity_mg;
    int minor_activity_eg;
    int tempo_mg;
    int tempo_eg;
} CPUEvalWeights;

typedef struct {
    double exploitation_weight;
    double exploration_weight;
    double prior_weight;
    double check_weight;
    double variance_weight;
    double visit_bias;
    double implicit_minimax_weight;
} CPUPolicyParams;

typedef struct {
    int quiescence_depth;
    int reply_search_depth;
    int sample_size;
    double randomness;
    int tactical_reply_threshold;
    double tactical_score_scale;
} CPURolloutParams;

typedef struct {
    int simulations;
    int playout_depth;
    int alpha_beta_depth;
    int thread_count;
} CPUSearchSettings;

void CPUSetEvalWeights(const CPUEvalWeights *weights);
void CPUGetEvalWeights(CPUEvalWeights *weights);
void CPUSetPolicyParams(const CPUPolicyParams *params);
void CPUGetPolicyParams(CPUPolicyParams *params);
void CPUSetRolloutParams(const CPURolloutParams *params);
void CPUGetRolloutParams(CPURolloutParams *params);
void CPUSetSearchSettings(const CPUSearchSettings *settings);
void CPUGetSearchSettings(CPUSearchSettings *settings);
void CPUTreeSearch(Board currentState, char player, const char *difficultyMode,
                      char *bestMove, size_t bestMoveSize);

#endif

