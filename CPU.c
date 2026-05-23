/*
    The algorithm's presented come from a variety of sources which
    were all glued together and heavily modified to create the final result. 
    The main inspirations and sources of ideas include:
    https://arxiv.org/abs/2006.08785
    "POLICY IMPROVEMENT BY PLANNING WITH GUMBEL"
    (used a little less but one idea I liked was to not ranked soley on how many visits for a move)
    https://www.chessprogramming.org/Quiescence_Search
    https://www.chessprogramming.org/MVV-LVA


    https://mlanctot.info/files/papers/cig14-immcts.pdf
    "Monte Carlo Tree Search with Heuristic Evaluations using Implicit Minimax Backups"

    "How do multiple threads search without all wasting time on the same move?"




*/


#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "Timer.h"

#include "Board.h"
#include "Piece.h"
#include "rules.h"
#include "CPU.h"

#define CPU_MAX_SQUARES (COLUMN * ROW)
#define CPU_MAX_MOVES (CPU_MAX_SQUARES * CPU_MAX_SQUARES)

#define CPU_EASY_SIMULATIONS 256
#define CPU_MEDIUM_SIMULATIONS 512
#define CPU_HARD_SIMULATIONS 2048
#define CPU_EASY_DEPTH 1
#define CPU_MEDIUM_DEPTH 1
#define CPU_HARD_DEPTH 1
#define CPU_EASY_ALPHA_BETA_DEPTH 2
#define CPU_MEDIUM_ALPHA_BETA_DEPTH 2
#define CPU_HARD_ALPHA_BETA_DEPTH 3
#define CPU_EASY_THREADS 16
#define CPU_MEDIUM_THREADS 32
#define CPU_HARD_THREADS 64

#define CPU_DEFAULT_QUIESCENCE_DEPTH 1 //How many plies to search in quiescence search before giving up and returning the static evaluation of the position
#define CPU_DEFAULT_ROLLOUT_REPLY_DEPTH 0 //How many plies to search for a reply in the rollout phase before giving up and just doing a static evaluation
#define CPU_THREADS_MAX 64
#define CPU_DEFAULT_ROLLOUT_SAMPLE_SIZE 15
#define CPU_DEFAULT_ROLLOUT_RANDOMNESS 0.03897875025196648
#define CPU_DEFAULT_ROLLOUT_REPLY_THRESHOLD -304
#define CPU_DEFAULT_ROLLOUT_TACTICAL_SCALE 5318.3900171203895
#define CPU_INITIAL_CHILD_CAPACITY 8
#define CPU_ROOT_TACTICAL_WEIGHT 0.87
#define CPU_CAPTURE_PRIOR_VISITS 0
#define CPU_MAX_PIECES 40

typedef struct {
    Board board;
    Piece pieces[CPU_MAX_PIECES];
    unsigned char alive[CPU_MAX_PIECES];
} CPUBoard;

typedef struct NewTreeNode {
    CPUBoard state;
    struct NewTreeNode *parent;
    struct NewTreeNode **children;
    int child_count;
    int child_capacity;
    int *untried_moves;
    int untried_count;
    int untried_total;
    int moves_generated;
    int visits;
    double total_value;
    double total_square_value;
    double heuristic_value;
    double implicit_minimax_value;
    double check_pressure;
    int heuristic_ready;
    int move_that_led_here_encoded;
    char move_that_led_here[10];
    char player;
} NewTreeNode;

typedef struct {
    double exploitation_weight;
    double exploration_weight;
    double prior_weight;
    double check_weight;
    double variance_weight;
    double visit_bias;
    double implicit_minimax_weight;
} NewSelectionPolicy;

typedef struct {
    int simulations;
    int playout_depth;
    int alpha_beta_depth;
    int thread_count;
} CPUSettings;

typedef struct {
    int move;
    CPUBoard state;
    char next_player;
    NewTreeNode *subroot;
    int terminal;
    int in_flight;
    int tree_mutex_ready; // Flag to indicate if the tree mutex has been initialized and is ready for use
    pthread_mutex_t tree_mutex; // Mutex to protect access to the search tree during updates
    int prior_visits;
    int visits;
    double prior_total_value;
    double prior_total_square_value;
    double total_value;
    double total_square_value;
    double heuristic_value;
    double tactical_value;
    double implicit_minimax_value;
    double check_pressure;
} NewRootChild;

typedef struct {
    NewRootChild *children;
    int child_count;
    int simulation_budget;
    int simulations_reserved;
    int simulations_completed;
    int active_jobs;
    const NewSelectionPolicy *policy;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
} NewRootScheduler;

typedef struct {
    NewRootScheduler *scheduler;
    unsigned int seed;
} NewSearchWorkerTask;

typedef struct {
    Piece *moving_piece;
    Position from_position;
    Position to_position;
    Type original_type;
    int original_has_moved;
    Piece *castling_rook;
    Position castling_rook_from_position;
    Position castling_rook_to_position;
    int castling_rook_original_has_moved;
    Piece *captured_pieces[CPU_MAX_SQUARES];
    Position captured_positions[CPU_MAX_SQUARES];
    int capture_count;
    int original_en_passant_available;
    Position original_en_passant_target;
    Position original_en_passant_pawn;
    Color original_en_passant_victim_color;
} CPUMoveUndo;

static char CPU_starting_player = 'W';
static CPUSettings CPU_settings = {
    CPU_MEDIUM_SIMULATIONS,
    CPU_MEDIUM_DEPTH,
    CPU_MEDIUM_ALPHA_BETA_DEPTH,
    CPU_MEDIUM_THREADS
};
static CPUEvalWeights CPU_eval_weights = {
    101, 351, 325, 500, 950, 191,
    170, 453, 325, 520, 1000, 131,
    12, 2, 0, 12, 2, 9,
    8, 8, 11, 36, 17, 18,
    8, 7, 16, 12, 3, 0,
    8, 4
};
static CPUPolicyParams CPU_policy_params = {
    1.5,
    0.8791211456010269,
    -0.5,
    -0.46417737324011477,
    0.38620775099963106,
    -0.7264038848079722,
    0.3632597256175234,
};
static CPURolloutParams CPU_rollout_params = {
    CPU_DEFAULT_QUIESCENCE_DEPTH,
    CPU_DEFAULT_ROLLOUT_REPLY_DEPTH,
    CPU_DEFAULT_ROLLOUT_SAMPLE_SIZE,
    CPU_DEFAULT_ROLLOUT_RANDOMNESS,
    CPU_DEFAULT_ROLLOUT_REPLY_THRESHOLD,
    CPU_DEFAULT_ROLLOUT_TACTICAL_SCALE
};
static _Thread_local unsigned int CPU_rng_state = 0; 
//Thread-local state for the random number generator, ensuring that each thread has its own independent sequence of random numbers.
//Pretty cool imo


static int captured_piece_value_for_move(Board *state, int move);
static int piece_controls_square(Board *state, Piece *piece, Position target_position);

void CPUSetEvalWeights(const CPUEvalWeights *weights)
{
    if (weights != NULL) {
        CPU_eval_weights = *weights;
    }
}

void CPUGetEvalWeights(CPUEvalWeights *weights)
{
    if (weights != NULL) {
        *weights = CPU_eval_weights;
    }
}

void CPUSetPolicyParams(const CPUPolicyParams *params)
{
    if (params != NULL) {
        CPU_policy_params = *params;
    }
}

void CPUGetPolicyParams(CPUPolicyParams *params)
{
    if (params != NULL) {
        *params = CPU_policy_params;
    }
}

void CPUSetRolloutParams(const CPURolloutParams *params)
{
    if (params == NULL) {
        return;
    }

    CPU_rollout_params = *params;
    if (CPU_rollout_params.quiescence_depth < 0) {
        CPU_rollout_params.quiescence_depth = 0;
    }
    if (CPU_rollout_params.quiescence_depth > 8) {
        CPU_rollout_params.quiescence_depth = 8;
    }
    if (CPU_rollout_params.reply_search_depth < 0) {
        CPU_rollout_params.reply_search_depth = 0;
    }
    if (CPU_rollout_params.reply_search_depth > 8) {
        CPU_rollout_params.reply_search_depth = 8;
    }
    if (CPU_rollout_params.sample_size < 0) {
        CPU_rollout_params.sample_size = 0;
    }
    if (CPU_rollout_params.sample_size > CPU_MAX_MOVES) {
        CPU_rollout_params.sample_size = CPU_MAX_MOVES;
    }
    if (CPU_rollout_params.randomness < 0.0) {
        CPU_rollout_params.randomness = 0.0;
    }
    if (CPU_rollout_params.randomness > 2.0) {
        CPU_rollout_params.randomness = 2.0;
    }
    if (CPU_rollout_params.tactical_score_scale < 1.0) {
        CPU_rollout_params.tactical_score_scale = 1.0;
    }
}

void CPUGetRolloutParams(CPURolloutParams *params)
{
    if (params != NULL) {
        *params = CPU_rollout_params;
    }
}

void CPUSetSearchSettings(const CPUSearchSettings *settings)
{
    if (settings == NULL) {
        return;
    }

    CPU_settings.simulations = settings->simulations;
    CPU_settings.playout_depth = settings->playout_depth;
    CPU_settings.alpha_beta_depth = settings->alpha_beta_depth;
    CPU_settings.thread_count = settings->thread_count;

    if (CPU_settings.simulations < 1) {
        CPU_settings.simulations = 1;
    }
    if (CPU_settings.playout_depth < 0) {
        CPU_settings.playout_depth = 0;
    }
    if (CPU_settings.alpha_beta_depth < 0) {
        CPU_settings.alpha_beta_depth = 0;
    }
    if (CPU_settings.thread_count < 1) {
        CPU_settings.thread_count = 1;
    }
    if (CPU_settings.thread_count > CPU_THREADS_MAX) {
        CPU_settings.thread_count = CPU_THREADS_MAX;
    }
}

void CPUGetSearchSettings(CPUSearchSettings *settings)
{
    if (settings != NULL) {
        settings->simulations = CPU_settings.simulations;
        settings->playout_depth = CPU_settings.playout_depth;
        settings->alpha_beta_depth = CPU_settings.alpha_beta_depth;
        settings->thread_count = CPU_settings.thread_count;
    }
}

static void set_thread_seed(unsigned int seed)
{
    CPU_rng_state = seed != 0 ? seed : 2463534242u;
}

static unsigned int next_random_u32(void)
{
    unsigned int x = CPU_rng_state;

    if (x == 0) {
        x = (unsigned int)time(NULL) ^ 0x9e3779b9u;
    }

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    CPU_rng_state = x;

    return x;
}

static char normalize_player(char player)
{
    return (player == 'B' || player == 'b') ? 'B' : 'W';
}


static char opponent_player(char player)
{
    return normalize_player(player) == 'W' ? 'B' : 'W';
}


static Color player_color(char player)
{
    return normalize_player(player) == 'W' ? WHITE : BLACK;
}

static int piece_belongs_to_player(const Piece *piece, char player)
{
    return piece != NULL && piece->color == player_color(player);
}

static double random_unit(void)
{
    return (double)next_random_u32() / 4294967295.0;
}

// Clamps a double value between a specified minimum and maximum, returning the clamped value.
static double clamp_double(double value, double min_value, double max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static int piece_phase_weight(Type type)
{
    switch (type) {
        case KNIGHT:
        case BISHOP:
        case ANTEATER:
            return 1;
        case ROOK:
            return 2;
        case QUEEN:
            return 4;
        default:
            return 0;
    }
}

static int piece_value_for_phase(Type type, int endgame)
{
    switch (type) {
        case PAWN:
            return endgame ? CPU_eval_weights.pawn_eg : CPU_eval_weights.pawn_mg;
        case KNIGHT:
            return endgame ? CPU_eval_weights.knight_eg : CPU_eval_weights.knight_mg;
        case BISHOP:
            return endgame ? CPU_eval_weights.bishop_eg : CPU_eval_weights.bishop_mg;
        case ANTEATER:
            return endgame ? CPU_eval_weights.anteater_eg : CPU_eval_weights.anteater_mg;
        case ROOK:
            return endgame ? CPU_eval_weights.rook_eg : CPU_eval_weights.rook_mg;
        case QUEEN:
            return endgame ? CPU_eval_weights.queen_eg : CPU_eval_weights.queen_mg;
        case KING:
        default:
            return 0;
    }
}

// Encodes a move from one position to another as a single integer, using the formula: move = from * (number of squares) + to.
static int encode_move(Position from, Position to) 
{
    return (int)from * CPU_MAX_SQUARES + (int)to;
}

static void decode_move(int move, int *from_col, int *from_row, int *to_col, int *to_row)
{
    int from = move / CPU_MAX_SQUARES;
    int to = move % CPU_MAX_SQUARES;

    *from_col = from / ROW;
    *from_row = from % ROW;
    *to_col = to / ROW;
    *to_row = to % ROW;
}

static void move_to_string(int move, char *out, size_t out_size)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;

    if (out == NULL || out_size == 0) {
        return;
    }

    decode_move(move, &from_col, &from_row, &to_col, &to_row);
    snprintf(out, out_size, "%c%d%c%d",
             'A' + from_col, from_row + 1, 'A' + to_col, to_row + 1);
}

static int is_inside_board_coords(int col, int row)
{
    return col >= 0 && col < COLUMN && row >= 0 && row < ROW;
}

static Position position_from_coords(int col, int row)
{
    return (Position)(col * ROW + row);
}

static void clear_en_passant_state(Board *board)
{
    if (board == NULL) {
        return;
    }

    board->en_passant_available = 0;
    board->en_passant_target = A1;
    board->en_passant_pawn = A1;
    board->en_passant_victim_color = WHITE;
}

static void clear_search_board(CPUBoard *state)
{
    if (state == NULL) {
        return;
    }

    for (int i = 0; i < CPU_MAX_PIECES; i++) {
        state->alive[i] = 0;
    }
    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            state->board.square[col][row] = NULL;
        }
    }
    clear_en_passant_state(&state->board);
}

// Finds the index of a piece in the CPUBoard's pieces array, returning -1 if the piece is not found or if the state or piece is NULL.
static int search_piece_slot_index(const CPUBoard *state, const Piece *piece)
{
    ptrdiff_t index; //Stores result of subtracting two pointers???

    if (state == NULL || piece == NULL) {
        return -1;
    }

    index = piece - state->pieces;
    if (index < 0 || index >= CPU_MAX_PIECES) {
        return -1;
    }
    return (int)index;
}

static void rebuild_search_board(CPUBoard *state)
{
    if (state == NULL) {
        return;
    }

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            state->board.square[col][row] = NULL;
        }
    }

    for (int i = 0; i < CPU_MAX_PIECES; i++) {
        if (state->alive[i]) {
            int col = FindColumnPosition(state->pieces[i].position);
            int row = FindRowPosition(state->pieces[i].position);

            if (is_inside_board_coords(col, row)) {
                state->board.square[col][row] = &state->pieces[i];
            }
        }
    }
}

//
static void init_search_board_from_board(CPUBoard *destination, const Board *source)
{
    int next_slot = 0;

    if (destination == NULL || source == NULL) {
        clear_search_board(destination);
        return;
    }

    clear_search_board(destination);
    destination->board.en_passant_available = source->en_passant_available;
    destination->board.en_passant_target = source->en_passant_target;
    destination->board.en_passant_pawn = source->en_passant_pawn;
    destination->board.en_passant_victim_color = source->en_passant_victim_color;

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = source->square[col][row];
            if (piece != NULL && next_slot < CPU_MAX_PIECES) {
                destination->pieces[next_slot] = *piece;
                destination->alive[next_slot] = 1;
                destination->board.square[col][row] = &destination->pieces[next_slot];
                next_slot++;
            }
        }
    }
}

static void copy_search_board(CPUBoard *destination, const CPUBoard *source)
{
    if (destination == NULL) {
        return;
    }

    memset(destination, 0, sizeof(*destination)); //Zero out the destination to ensure any padding bytes are cleared and alive flags are reset
    if (source == NULL) {
        clear_search_board(destination);
        return;
    }

    *destination = *source;
    rebuild_search_board(destination);
}

// Adds a candidate move to the moves array if it's valid and there's room, checking for collisions and move legality.
static void add_candidate_move(Board *state, Piece *piece, int target_col,
                               int target_row, int *moves, int max_moves,
                               int *count)
{
    Piece *target;
    Position target_position;

    if (state == NULL || piece == NULL || moves == NULL || count == NULL ||
        !is_inside_board_coords(target_col, target_row)) {
        return;
    }

    target = state->square[target_col][target_row];
    if (target != NULL && target->color == piece->color) {
        return;
    }

    target_position = position_from_coords(target_col, target_row);
    if (*count < max_moves &&
        IsValidMove(state, piece, target_position)) {
        moves[(*count)++] = encode_move(piece->position, target_position);
    }
}

static void add_capture_candidate_move(Board *state, Piece *piece, int target_col,
                                       int target_row, int *moves, int max_moves,
                                       int *count)
{
    Piece *target;
    Position target_position;
    int is_en_passant_capture = 0;

    if (state == NULL || piece == NULL || moves == NULL || count == NULL ||
        !is_inside_board_coords(target_col, target_row)) {
        return;
    }

    target = state->square[target_col][target_row];
    target_position = position_from_coords(target_col, target_row);

    if (target != NULL) {
        if (target->color == piece->color) {
            return;
        }
    } else {
        is_en_passant_capture =
            piece->type == PAWN &&
            state->en_passant_available &&
            target_position == state->en_passant_target;
        if (!is_en_passant_capture) {
            return;
        }
    }

    if (*count < max_moves &&
        IsValidMove(state, piece, target_position)) {
        moves[(*count)++] = encode_move(piece->position, target_position);
    }
}

// For sliding pieces, adds all valid moves in the given directions until blocked by another piece or the edge of the board.
static void add_sliding_moves(Board *state, Piece *piece, int from_col, int from_row,
                              const int directions[][2], int direction_count,
                              int *moves, int max_moves, int *count)
{
    for (int direction = 0; direction < direction_count; direction++) {
        int col = from_col + directions[direction][0];
        int row = from_row + directions[direction][1];

        while (is_inside_board_coords(col, row)) {
            Piece *target = state->square[col][row];

            if (target != NULL && target->color == piece->color) {
                break;
            }

            add_candidate_move(state, piece, col, row, moves, max_moves, count);
            if (target != NULL) {
                break;
            }

            col += directions[direction][0];
            row += directions[direction][1];
        }
    }
}

static void add_sliding_capture_moves(Board *state, Piece *piece, int from_col, int from_row,
                                      const int directions[][2], int direction_count,
                                      int *moves, int max_moves, int *count)
{
    for (int direction = 0; direction < direction_count; direction++) {
        int col = from_col + directions[direction][0];
        int row = from_row + directions[direction][1];

        while (is_inside_board_coords(col, row)) {
            Piece *target = state->square[col][row];

            if (target != NULL) {
                if (target->color != piece->color) {
                    add_capture_candidate_move(state, piece, col, row,
                                               moves, max_moves, count);
                }
                break;
            }

            col += directions[direction][0];
            row += directions[direction][1];
        }
    }
}

static int generate_valid_moves(Board *state, char player, int *moves, int max_moves)
{
    int count = 0;
    char current_player = normalize_player(player);
    static const int knight_offsets[8][2] = {
        { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 },
        { -1, -2 }, { -2, -1 }, { -2, 1 }, { -1, 2 }
    };
    static const int adjacent_offsets[8][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 },
        { 0, -1 },             { 0, 1 },
        { 1, -1 },  { 1, 0 },  { 1, 1 }
    };
    static const int rook_directions[4][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }
    };
    static const int bishop_directions[4][2] = {
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };
    static const int queen_directions[8][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };

    if (state == NULL || moves == NULL || max_moves <= 0) {
        return 0;
    }

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = state->square[col][row];

            if (!piece_belongs_to_player(piece, current_player)) {
                continue;
            }

            switch (piece->type) {
                case PAWN: {
                    int direction = piece->color == WHITE ? 1 : -1;
                    int forward_row = row + direction;
                    int start_row = piece->color == WHITE ? 1 : ROW - 2;

                    if (is_inside_board_coords(col, forward_row) &&
                        state->square[col][forward_row] == NULL) {
                        add_candidate_move(state, piece, col, forward_row,
                                           moves, max_moves, &count);

                        if (row == start_row) {
                            int double_row = row + 2 * direction;
                            if (is_inside_board_coords(col, double_row) &&
                                state->square[col][double_row] == NULL) {
                                add_candidate_move(state, piece, col, double_row,
                                                   moves, max_moves, &count);
                            }
                        }
                    }

                    for (int offset = -1; offset <= 1; offset += 2) {
                        int target_col = col + offset;
                        int target_row = row + direction;
                        Piece *target;
                        Position target_position;

                        if (!is_inside_board_coords(target_col, target_row)) {
                            continue;
                        }

                        target = state->square[target_col][target_row];
                        target_position = position_from_coords(target_col, target_row);
                        if ((target != NULL && target->color != piece->color) ||
                            (target == NULL &&
                             state->en_passant_available &&
                             target_position == state->en_passant_target)) {
                            add_candidate_move(state, piece, target_col, target_row,
                                               moves, max_moves, &count);
                        }
                    }
                    break;
                }

                case KNIGHT:
                    for (int i = 0; i < 8; i++) {
                        add_candidate_move(state, piece,
                                           col + knight_offsets[i][0],
                                           row + knight_offsets[i][1],
                                           moves, max_moves, &count);
                    }
                    break;

                case BISHOP:
                    add_sliding_moves(state, piece, col, row, bishop_directions, 4,
                                      moves, max_moves, &count);
                    break;

                case ROOK:
                    add_sliding_moves(state, piece, col, row, rook_directions, 4,
                                      moves, max_moves, &count);
                    break;

                case QUEEN:
                    add_sliding_moves(state, piece, col, row, queen_directions, 8,
                                      moves, max_moves, &count);
                    break;

                case KING:
                    for (int i = 0; i < 8; i++) {
                        add_candidate_move(state, piece,
                                           col + adjacent_offsets[i][0],
                                           row + adjacent_offsets[i][1],
                                           moves, max_moves, &count);
                    }
                    add_candidate_move(state, piece, col - 2, row, moves, max_moves, &count);
                    add_candidate_move(state, piece, col + 2, row, moves, max_moves, &count);
                    break;

                case ANTEATER:
                    for (int target_col = 0; target_col < COLUMN; target_col++) {
                        for (int target_row = 0; target_row < ROW; target_row++) {
                            add_candidate_move(state, piece, target_col, target_row,
                                               moves, max_moves, &count);
                        }
                    }
                    break;

                default:
                    break;
            }
        }
    }

    if (count == 0) {
        return 0;
    }

    return count;
}

static int generate_capture_moves(Board *state, char player, int *moves, int max_moves)
{
    int count = 0;
    char current_player = normalize_player(player);
    static const int knight_offsets[8][2] = {
        { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 },
        { -1, -2 }, { -2, -1 }, { -2, 1 }, { -1, 2 }
    };
    static const int adjacent_offsets[8][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 },
        { 0, -1 },             { 0, 1 },
        { 1, -1 },  { 1, 0 },  { 1, 1 }
    };
    static const int rook_directions[4][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }
    };
    static const int bishop_directions[4][2] = {
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };
    static const int queen_directions[8][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };

    if (state == NULL || moves == NULL || max_moves <= 0) {
        return 0;
    }

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = state->square[col][row];

            if (!piece_belongs_to_player(piece, current_player)) {
                continue;
            }

            switch (piece->type) {
                case PAWN: {
                    int direction = piece->color == WHITE ? 1 : -1;

                    for (int offset = -1; offset <= 1; offset += 2) {
                        int target_col = col + offset;
                        int target_row = row + direction;

                        add_capture_candidate_move(state, piece, target_col, target_row,
                                                   moves, max_moves, &count);
                    }
                    break;
                }

                case KNIGHT:
                    for (int i = 0; i < 8; i++) {
                        add_capture_candidate_move(state, piece,
                                                   col + knight_offsets[i][0],
                                                   row + knight_offsets[i][1],
                                                   moves, max_moves, &count);
                    }
                    break;

                case BISHOP:
                    add_sliding_capture_moves(state, piece, col, row,
                                              bishop_directions, 4,
                                              moves, max_moves, &count);
                    break;

                case ROOK:
                    add_sliding_capture_moves(state, piece, col, row,
                                              rook_directions, 4,
                                              moves, max_moves, &count);
                    break;

                case QUEEN:
                    add_sliding_capture_moves(state, piece, col, row,
                                              queen_directions, 8,
                                              moves, max_moves, &count);
                    break;

                case KING:
                    for (int i = 0; i < 8; i++) {
                        add_capture_candidate_move(state, piece,
                                                   col + adjacent_offsets[i][0],
                                                   row + adjacent_offsets[i][1],
                                                   moves, max_moves, &count);
                    }
                    break;

                case ANTEATER:
                    for (int target_col = 0; target_col < COLUMN; target_col++) {
                        for (int target_row = 0; target_row < ROW; target_row++) {
                            Piece *target = state->square[target_col][target_row];

                            if (target == NULL || target->color == piece->color) {
                                continue;
                            }

                            add_capture_candidate_move(state, piece, target_col, target_row,
                                                       moves, max_moves, &count);
                        }
                    }
                    break;

                default:
                    break;
            }
        }
    }

    return count;
}

static int *get_valid_moves(Board *state, char player, int *move_count)
{
    int generated_moves[CPU_MAX_MOVES];
    int *moves;
    int count;

    if (move_count != NULL) {
        *move_count = 0;
    }

    count = generate_valid_moves(state, player, generated_moves, CPU_MAX_MOVES);
    if (count <= 0) {
        return NULL;
    }

    moves = (int *)malloc(sizeof(int) * (size_t)count);
    if (moves == NULL) {
        return NULL;
    }

    memcpy(moves, generated_moves, sizeof(int) * (size_t)count); //Copy the generated moves from the local array to the dynamically allocated array that will be returned to the caller.
    if (move_count != NULL) {
        *move_count = count;
    }

    return moves;
}

static int has_king(const Board *state, char player)
{
    Color color = player_color(player);

    if (state == NULL) {
        return 0;
    }

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = state->square[col][row];
            if (piece != NULL && piece->color == color && piece->type == KING) {
                return 1;
            }
        }
    }

    return 0;
}

static int color_index(Color color)
{
    return color == WHITE ? 0 : 1;
}

static int perspective_row_for_color(int row, Color color)
{
    return color == WHITE ? row : (ROW - 1 - row);
}

// Returns a bonus score based on how close the square is to the center of the board, with a maximum of 14 for the central square and decreasing towards the edges.
static int center_bonus_for_square(int col, int row)
{
    int file_distance = abs(2 * col - (COLUMN - 1));
    int rank_distance = abs(2 * row - (ROW - 1));
    int bonus = 14 - file_distance - rank_distance;

    return bonus > 0 ? bonus : 0;
}

//Checks if there is a pawn of the specified color in the given file (column) of the board, returning 1 if found and 0 otherwise.
static int file_has_pawn(Board *state, Color color, int file) 
{
    if (state == NULL || file < 0 || file >= COLUMN) {
        return 0;
    }

    for (int row = 0; row < ROW; row++) {
        Piece *piece = state->square[file][row];
        if (piece != NULL && piece->color == color && piece->type == PAWN) {
            return 1;
        }
    }

    return 0;
}

static int is_passed_pawn(Board *state, Piece *piece, int col, int row)
{
    int direction;
    Color opponent;

    if (state == NULL || piece == NULL || piece->type != PAWN) {
        return 0;
    }

    direction = piece->color == WHITE ? 1 : -1;
    opponent = piece->color == WHITE ? BLACK : WHITE;

    for (int file = col - 1; file <= col + 1; file++) {
        if (file < 0 || file >= COLUMN) {
            continue;
        }

        for (int scan_row = row + direction;
             scan_row >= 0 && scan_row < ROW;
             scan_row += direction) {
            Piece *blocker = state->square[file][scan_row];
            if (blocker != NULL && blocker->color == opponent && blocker->type == PAWN) {
                return 0;
            }
        }
    }

    return 1;
}

static int sliding_mobility(Board *state, Piece *piece, int from_col, int from_row,
                            const int directions[][2], int direction_count)
{
    int mobility = 0;

    for (int direction = 0; direction < direction_count; direction++) {
        int col = from_col + directions[direction][0];
        int row = from_row + directions[direction][1];

        while (is_inside_board_coords(col, row)) {
            Piece *target = state->square[col][row];

            if (target != NULL && target->color == piece->color) {
                break;
            }

            mobility++;
            if (target != NULL) {
                break;
            }

            col += directions[direction][0];
            row += directions[direction][1];
        }
    }

    return mobility;
}

static int piece_mobility(Board *state, Piece *piece, int col, int row)
{
    int mobility = 0;
    static const int knight_offsets[8][2] = {
        { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 },
        { -1, -2 }, { -2, -1 }, { -2, 1 }, { -1, 2 }
    };
    static const int adjacent_offsets[8][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 },
        { 0, -1 },             { 0, 1 },
        { 1, -1 },  { 1, 0 },  { 1, 1 }
    };
    static const int rook_directions[4][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }
    };
    static const int bishop_directions[4][2] = {
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };
    static const int queen_directions[8][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };

    if (state == NULL || piece == NULL) {
        return 0;
    }

    switch (piece->type) {
        case PAWN: {
            int direction = piece->color == WHITE ? 1 : -1;
            if (is_inside_board_coords(col, row + direction) &&
                state->square[col][row + direction] == NULL) {
                mobility++;
            }
            for (int offset = -1; offset <= 1; offset += 2) {
                int target_col = col + offset;
                int target_row = row + direction;
                if (is_inside_board_coords(target_col, target_row) &&
                    state->square[target_col][target_row] != NULL &&
                    state->square[target_col][target_row]->color != piece->color) {
                    mobility++;
                }
            }
            break;
        }

        case KNIGHT:
            for (int i = 0; i < 8; i++) {
                int target_col = col + knight_offsets[i][0];
                int target_row = row + knight_offsets[i][1];
                Piece *target;
                if (!is_inside_board_coords(target_col, target_row)) {
                    continue;
                }
                target = state->square[target_col][target_row];
                if (target == NULL || target->color != piece->color) {
                    mobility++;
                }
            }
            break;

        case BISHOP:
            mobility = sliding_mobility(state, piece, col, row, bishop_directions, 4);
            break;

        case ROOK:
            mobility = sliding_mobility(state, piece, col, row, rook_directions, 4);
            break;

        case QUEEN:
            mobility = sliding_mobility(state, piece, col, row, queen_directions, 8);
            break;

        case KING:
            for (int i = 0; i < 8; i++) {
                int target_col = col + adjacent_offsets[i][0];
                int target_row = row + adjacent_offsets[i][1];
                Piece *target;
                if (!is_inside_board_coords(target_col, target_row)) {
                    continue;
                }
                target = state->square[target_col][target_row];
                if (target == NULL || target->color != piece->color) {
                    mobility++;
                }
            }
            break;

        case ANTEATER:
            for (int target_col = 0; target_col < COLUMN; target_col++) {
                for (int target_row = 0; target_row < ROW; target_row++) {
                    Piece *target = state->square[target_col][target_row];
                    Position target_position = position_from_coords(target_col, target_row);

                    if (target_col == col && target_row == row) {
                        continue;
                    }

                    if (target == NULL) {
                        if (abs(target_col - col) <= 1 &&
                            abs(target_row - row) <= 1) {
                            mobility++;
                        }
                    } else if (target->color != piece->color &&
                               target->type == PAWN &&
                               FindAnteaterCapturePath(state, piece,
                                                       target_position, NULL, 0) > 0) {
                        mobility++;
                    }
                }
            }
            break;

        default:
            break;
    }

    return mobility;
}

static int threat_units_for_target(Piece *attacker, Piece *target)
{
    if (attacker == NULL || target == NULL || attacker->color == target->color) {
        return 0;
    }

    return piece_value_for_phase(target->type, 0) / 100;
}

static int threat_units_for_piece(Board *state, Piece *piece, int col, int row)
{
    int threats = 0;
    static const int knight_offsets[8][2] = {
        { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 },
        { -1, -2 }, { -2, -1 }, { -2, 1 }, { -1, 2 }
    };
    static const int adjacent_offsets[8][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 },
        { 0, -1 },             { 0, 1 },
        { 1, -1 },  { 1, 0 },  { 1, 1 }
    };
    static const int rook_directions[4][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }
    };
    static const int bishop_directions[4][2] = {
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };
    static const int queen_directions[8][2] = {
        { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }
    };
    const int (*directions)[2] = NULL;
    int direction_count = 0;

    if (state == NULL || piece == NULL) {
        return 0;
    }

    switch (piece->type) {
        case PAWN: {
            int direction = piece->color == WHITE ? 1 : -1;
            for (int offset = -1; offset <= 1; offset += 2) {
                int target_col = col + offset;
                int target_row = row + direction;
                if (is_inside_board_coords(target_col, target_row)) {
                    threats += threat_units_for_target(piece,
                        state->square[target_col][target_row]);
                }
            }
            return threats;
        }

        case KNIGHT:
            for (int i = 0; i < 8; i++) {
                int target_col = col + knight_offsets[i][0];
                int target_row = row + knight_offsets[i][1];
                if (is_inside_board_coords(target_col, target_row)) {
                    threats += threat_units_for_target(piece,
                        state->square[target_col][target_row]);
                }
            }
            return threats;

        case KING:
            for (int i = 0; i < 8; i++) {
                int target_col = col + adjacent_offsets[i][0];
                int target_row = row + adjacent_offsets[i][1];
                Piece *target;
                if (!is_inside_board_coords(target_col, target_row)) {
                    continue;
                }
                target = state->square[target_col][target_row];
                threats += threat_units_for_target(piece, target);
            }
            return threats;

        case ANTEATER:
            for (int target_col = 0; target_col < COLUMN; target_col++) {
                for (int target_row = 0; target_row < ROW; target_row++) {
                    Piece *target = state->square[target_col][target_row];
                    Position target_position = position_from_coords(target_col, target_row);

                    if (target != NULL && target->color != piece->color &&
                        target->type == PAWN &&
                        FindAnteaterCapturePath(state, piece,
                                                target_position, NULL, 0) > 0) {
                        threats += threat_units_for_target(piece, target);
                    }
                }
            }
            return threats;

        case BISHOP:
            directions = bishop_directions;
            direction_count = 4;
            break;

        case ROOK:
            directions = rook_directions;
            direction_count = 4;
            break;

        case QUEEN:
            directions = queen_directions;
            direction_count = 8;
            break;

        default:
            return 0;
    }

    for (int direction = 0; direction < direction_count; direction++) {
        int target_col = col + directions[direction][0];
        int target_row = row + directions[direction][1];

        while (is_inside_board_coords(target_col, target_row)) {
            Piece *target = state->square[target_col][target_row];
            if (target != NULL) {
                threats += threat_units_for_target(piece, target);
                break;
            }
            target_col += directions[direction][0];
            target_row += directions[direction][1];
        }
    }

    return threats;
}


//Thinking thinking

static int count_king_shield(Board *state, Color color, int king_col, int king_row)
{
    int shield = 0;
    int direction = color == WHITE ? 1 : -1;

    for (int file = king_col - 1; file <= king_col + 1; file++) {
        int row = king_row + direction;
        Piece *piece;
        if (!is_inside_board_coords(file, row)) {
            continue;
        }
        piece = state->square[file][row];
        if (piece != NULL && piece->color == color && piece->type == PAWN) {
            shield++;
        }
    }

    return shield;
}

static double static_evaluate_position_for_side(Board *state, char player, char side_to_move)
{
    int mg_score[2] = {0, 0};
    int eg_score[2] = {0, 0};
    int pawn_files[2][COLUMN];
    int phase = 0;
    int total_phase = 0;
    int endgame_phase;
    int white_final;
    int black_final;
    double score;

    if (state == NULL) {
        return 0.0;
    }
    if (!has_king(state, player)) {
        return -1.0;
    }
    if (!has_king(state, opponent_player(player))) {
        return 1.0;
    }

    memset(pawn_files, 0, sizeof(pawn_files));
    total_phase = 4 * piece_phase_weight(KNIGHT) +
                  4 * piece_phase_weight(BISHOP) +
                  4 * piece_phase_weight(ROOK) +
                  2 * piece_phase_weight(QUEEN) +
                  4 * piece_phase_weight(ANTEATER);

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = state->square[col][row];
            int index;
            int perspective_row;
            int center_bonus;
            int mobility;
            int threats;
            if (piece == NULL) {
                continue;
            }

            index = color_index(piece->color);
            perspective_row = perspective_row_for_color(row, piece->color);
            center_bonus = center_bonus_for_square(col, row);
            mobility = piece_mobility(state, piece, col, row);
            threats = threat_units_for_piece(state, piece, col, row);

            mg_score[index] += piece_value_for_phase(piece->type, 0);
            eg_score[index] += piece_value_for_phase(piece->type, 1);
            phase += piece_phase_weight(piece->type);

            mg_score[index] += CPU_eval_weights.mobility_mg * mobility;
            eg_score[index] += CPU_eval_weights.mobility_eg * mobility;
            mg_score[index] += CPU_eval_weights.threat_mg * threats;
            eg_score[index] += CPU_eval_weights.threat_eg * threats;

            if (piece->type == PAWN) {
                int doubled;
                int isolated;
                int passed;
                pawn_files[index][col]++;
                mg_score[index] += CPU_eval_weights.pawn_advance_mg * perspective_row;
                eg_score[index] += CPU_eval_weights.pawn_advance_eg * perspective_row;

                doubled = pawn_files[index][col] > 1;
                isolated = !file_has_pawn(state, piece->color, col - 1) &&
                           !file_has_pawn(state, piece->color, col + 1);
                passed = is_passed_pawn(state, piece, col, row);
                if (doubled) {
                    mg_score[index] -= CPU_eval_weights.pawn_structure_mg;
                    eg_score[index] -= CPU_eval_weights.pawn_structure_eg;
                }
                if (isolated) {
                    mg_score[index] -= CPU_eval_weights.pawn_structure_mg;
                    eg_score[index] -= CPU_eval_weights.pawn_structure_eg;
                }
                if (passed) {
                    mg_score[index] += CPU_eval_weights.passed_pawn_mg *
                                       (perspective_row + 1);
                    eg_score[index] += CPU_eval_weights.passed_pawn_eg *
                                       (perspective_row + 1);
                }
            } else if (piece->type == KING) {
                int shield = count_king_shield(state, piece->color, col, row);
                mg_score[index] += CPU_eval_weights.king_safety_mg * shield;
                mg_score[index] -= CPU_eval_weights.center_mg * center_bonus;
                eg_score[index] += CPU_eval_weights.king_activity_eg * center_bonus;
            } else {
                mg_score[index] += CPU_eval_weights.center_mg * center_bonus;
                eg_score[index] += CPU_eval_weights.center_eg * center_bonus;

                if (piece->type == ROOK) {
                    int own_pawn_on_file = file_has_pawn(state, piece->color, col);
                    int enemy_pawn_on_file = file_has_pawn(state,
                        piece->color == WHITE ? BLACK : WHITE, col);
                    if (!own_pawn_on_file) {
                        mg_score[index] += CPU_eval_weights.rook_activity_mg;
                        eg_score[index] += CPU_eval_weights.rook_activity_eg;
                        if (!enemy_pawn_on_file) {
                            mg_score[index] += CPU_eval_weights.rook_activity_mg;
                            eg_score[index] += CPU_eval_weights.rook_activity_eg;
                        }
                    }
                } else if (piece->type == BISHOP || piece->type == KNIGHT) {
                    mg_score[index] += CPU_eval_weights.minor_activity_mg * center_bonus;
                    eg_score[index] += CPU_eval_weights.minor_activity_eg * center_bonus;
                }
            }

        }
    }

    if (IsInCheck(state, WHITE)) {
        mg_score[color_index(WHITE)] -= CPU_eval_weights.king_safety_mg * 4;
        eg_score[color_index(WHITE)] -= CPU_eval_weights.threat_eg * 4;
    }
    if (IsInCheck(state, BLACK)) {
        mg_score[color_index(BLACK)] -= CPU_eval_weights.king_safety_mg * 4;
        eg_score[color_index(BLACK)] -= CPU_eval_weights.threat_eg * 4;
    }

    side_to_move = normalize_player(side_to_move);
    if (side_to_move == 'W') {
        mg_score[color_index(WHITE)] += CPU_eval_weights.tempo_mg;
        eg_score[color_index(WHITE)] += CPU_eval_weights.tempo_eg;
    } else {
        mg_score[color_index(BLACK)] += CPU_eval_weights.tempo_mg;
        eg_score[color_index(BLACK)] += CPU_eval_weights.tempo_eg;
    }

    if (total_phase <= 0) {
        endgame_phase = 256;
    } else {
        if (phase > total_phase) {
            phase = total_phase;
        }
        endgame_phase = ((total_phase - phase) * 256 + total_phase / 2) / total_phase;
    }

    white_final = (mg_score[color_index(WHITE)] * (256 - endgame_phase) +
                   eg_score[color_index(WHITE)] * endgame_phase) / 256;
    black_final = (mg_score[color_index(BLACK)] * (256 - endgame_phase) +
                   eg_score[color_index(BLACK)] * endgame_phase) / 256;

    score = (double)(white_final - black_final);
    if (normalize_player(player) == 'B') {
        score = -score;
    }

    score /= 6500.0;
    return clamp_double(score, -1.0, 1.0);
}

//Returns a score representing the pressure on the player's king, with positive values indicating more pressure and negative values indicating less pressure.
static double check_pressure_for_player(Board *state, char player) 
{
    double pressure = 0.0;

    if (state == NULL) {
        return 0.0;
    }

    player = normalize_player(player);
    if (IsInCheck(state, player_color(opponent_player(player)))) {
        pressure += 1.0;
    }
    if (IsInCheck(state, player_color(player))) {
        pressure -= 1.0;
    }

    return pressure;
}

static double evaluate_position_for_side(Board *state, char player, char side_to_move)
{
    if (state == NULL) {
        return 0.0;
    }
    if (!has_king(state, player)) { 
        return -1.0;
    }
    if (!has_king(state, opponent_player(player))) {
        return 1.0;
    }

    return static_evaluate_position_for_side(state, player, side_to_move);
}

static void discard_move_undo_captures(CPUBoard *state, CPUMoveUndo *undo)
{
    if (undo == NULL) {
        return;
    }

    for (int i = 0; i < undo->capture_count; i++) {
        if (undo->captured_pieces[i] != NULL) {
            int slot_index = search_piece_slot_index(state, undo->captured_pieces[i]);
            if (slot_index >= 0) {
                state->alive[slot_index] = 0;
            }
            undo->captured_pieces[i] = NULL;
        }
    }
    undo->capture_count = 0;
}

static void undo_move_inplace(CPUBoard *state, const CPUMoveUndo *undo)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;

    if (state == NULL || undo == NULL || undo->moving_piece == NULL) {
        return;
    }

    from_col = FindColumnPosition(undo->from_position);
    from_row = FindRowPosition(undo->from_position);
    to_col = FindColumnPosition(undo->to_position);
    to_row = FindRowPosition(undo->to_position);

    if (is_inside_board_coords(to_col, to_row)) {
        state->board.square[to_col][to_row] = NULL;
    }
    if (undo->castling_rook != NULL) {
        int rook_from_col = FindColumnPosition(undo->castling_rook_from_position);
        int rook_from_row = FindRowPosition(undo->castling_rook_from_position);
        int rook_to_col = FindColumnPosition(undo->castling_rook_to_position);
        int rook_to_row = FindRowPosition(undo->castling_rook_to_position);

        if (is_inside_board_coords(rook_to_col, rook_to_row)) {
            state->board.square[rook_to_col][rook_to_row] = NULL;
        }
        if (is_inside_board_coords(rook_from_col, rook_from_row)) {
            state->board.square[rook_from_col][rook_from_row] = undo->castling_rook;
        }
        undo->castling_rook->position = undo->castling_rook_from_position;
        undo->castling_rook->has_moved = undo->castling_rook_original_has_moved;
    }
    if (is_inside_board_coords(from_col, from_row)) {
        state->board.square[from_col][from_row] = undo->moving_piece;
    }
    undo->moving_piece->position = undo->from_position;
    undo->moving_piece->type = undo->original_type;
    undo->moving_piece->has_moved = undo->original_has_moved;

    for (int i = 0; i < undo->capture_count; i++) {
        int capture_col = FindColumnPosition(undo->captured_positions[i]);
        int capture_row = FindRowPosition(undo->captured_positions[i]);
        Piece *captured_piece = undo->captured_pieces[i];
        int slot_index = search_piece_slot_index(state, captured_piece);

        if (captured_piece == NULL ||
            !is_inside_board_coords(capture_col, capture_row)) {
            continue;
        }

        if (slot_index >= 0) {
            state->alive[slot_index] = 1;
        }
        captured_piece->position = undo->captured_positions[i];
        state->board.square[capture_col][capture_row] = captured_piece;
    }

    state->board.en_passant_available = undo->original_en_passant_available;
    state->board.en_passant_target = undo->original_en_passant_target;
    state->board.en_passant_pawn = undo->original_en_passant_pawn;
    state->board.en_passant_victim_color = undo->original_en_passant_victim_color;
}


//Move execution and undo
static int make_move_inplace(CPUBoard *state, int move, char player,
                             CPUMoveUndo *undo, int validate_move,
                             int *is_terminal, double *reward)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;
    Board *board;
    Piece *moving_piece;
    Piece *captured_piece;
    Position target;
    int castling_move;
    int rook_from_col;
    int rook_to_col;
    Piece *castling_rook;
    int opponent_checkmated;

    if (is_terminal != NULL) {
        *is_terminal = 0;
    }
    if (reward != NULL) {
        *reward = 0.0;
    }
    if (state == NULL || undo == NULL) {
        return 0;
    }

    board = &state->board;
    memset(undo, 0, sizeof(*undo));

    decode_move(move, &from_col, &from_row, &to_col, &to_row);
    if (from_col < 0 || from_col >= COLUMN || from_row < 0 || from_row >= ROW ||
        to_col < 0 || to_col >= COLUMN || to_row < 0 || to_row >= ROW) {
        return 0;
    }

    moving_piece = board->square[from_col][from_row];
    captured_piece = board->square[to_col][to_row];
    target = MakePosition(to_col, to_row);
    castling_move = 0;
    rook_from_col = -1;
    rook_to_col = -1;
    castling_rook = NULL;

    if (!piece_belongs_to_player(moving_piece, player) ||
        piece_belongs_to_player(captured_piece, player) ||
        (validate_move && !IsValidMove(board, moving_piece, target))) {
        return 0;
    }

    undo->moving_piece = moving_piece;
    undo->from_position = moving_piece->position;
    undo->to_position = target;
    undo->original_type = moving_piece->type;
    undo->original_has_moved = moving_piece->has_moved;
    undo->original_en_passant_available = board->en_passant_available;
    undo->original_en_passant_target = board->en_passant_target;
    undo->original_en_passant_pawn = board->en_passant_pawn;
    undo->original_en_passant_victim_color = board->en_passant_victim_color;

    if (moving_piece->type == PAWN &&
        captured_piece == NULL &&
        board->en_passant_available &&
        target == board->en_passant_target) {
        int en_passant_col = FindColumnPosition(board->en_passant_pawn);
        int en_passant_row = FindRowPosition(board->en_passant_pawn);

        if (is_inside_board_coords(en_passant_col, en_passant_row) &&
            board->square[en_passant_col][en_passant_row] != NULL &&
            board->square[en_passant_col][en_passant_row]->type == PAWN &&
            board->square[en_passant_col][en_passant_row]->color != moving_piece->color) {
            undo->capture_count = 1;
            undo->captured_positions[0] = board->en_passant_pawn;
            undo->captured_pieces[0] = board->square[en_passant_col][en_passant_row];
            board->square[en_passant_col][en_passant_row] = NULL;
        }
    } else if (moving_piece->type == ANTEATER) {
        undo->capture_count = FindAnteaterCapturePath(board, moving_piece,
                                                      target,
                                                      undo->captured_positions,
                                                      CPU_MAX_SQUARES);
        for (int i = 0; i < undo->capture_count; i++) {
            int capture_col = FindColumnPosition(undo->captured_positions[i]);
            int capture_row = FindRowPosition(undo->captured_positions[i]);

            undo->captured_pieces[i] = board->square[capture_col][capture_row];
            board->square[capture_col][capture_row] = NULL;
        }
    } else if (moving_piece->type == KING &&
               from_row == to_row &&
               abs(to_col - from_col) == 2) {
        castling_move = 1;
        rook_from_col = to_col > from_col ? COLUMN - 1 : 0;
        rook_to_col = from_col + (to_col > from_col ? 1 : -1);
        castling_rook = board->square[rook_from_col][from_row];
        if (castling_rook == NULL ||
            castling_rook->type != ROOK ||
            castling_rook->color != moving_piece->color) {
            return 0;
        }
        undo->castling_rook = castling_rook;
        undo->castling_rook_from_position = castling_rook->position;
        undo->castling_rook_to_position = position_from_coords(rook_to_col, from_row);
        undo->castling_rook_original_has_moved = castling_rook->has_moved;
    } else if (captured_piece != NULL) {
        undo->capture_count = 1;
        undo->captured_positions[0] = target;
        undo->captured_pieces[0] = captured_piece;
        board->square[to_col][to_row] = NULL;
    }

    board->square[to_col][to_row] = moving_piece;
    board->square[from_col][from_row] = NULL;
    moving_piece->position = target;
    moving_piece->has_moved = 1;

    if (castling_move) {
        board->square[rook_to_col][from_row] = castling_rook;
        board->square[rook_from_col][from_row] = NULL;
        castling_rook->position = position_from_coords(rook_to_col, from_row);
        castling_rook->has_moved = 1;
    }

    clear_en_passant_state(board);
    if (moving_piece->type == PAWN && abs(to_row - from_row) == 2) {
        int direction = moving_piece->color == WHITE ? 1 : -1;
        board->en_passant_available = 1;
        board->en_passant_target = position_from_coords(from_col, from_row + direction);
        board->en_passant_pawn = target;
        board->en_passant_victim_color = moving_piece->color;
    }

    if (moving_piece->type == PAWN &&
        ((moving_piece->color == WHITE && to_row == ROW - 1) ||
         (moving_piece->color == BLACK && to_row == 0))) {
        moving_piece->type = QUEEN;
    }

    opponent_checkmated = IsCheckmate(board,
                                      player_color(opponent_player(player)));
    if (opponent_checkmated) {
        if (is_terminal != NULL) {
            *is_terminal = 1;
        }
        if (reward != NULL) {
            *reward = 1.0;
        }
    }

    return 1;
}

static int apply_move(const CPUBoard *state, int move, char player,
                      CPUBoard *next, int *is_terminal, double *reward)
{
    CPUMoveUndo undo;

    if (is_terminal != NULL) {
        *is_terminal = 0;
    }
    if (reward != NULL) {
        *reward = 0.0;
    }
    if (next == NULL) {
        return 0;
    }

    copy_search_board(next, state);

    if (!make_move_inplace(next, move, player, &undo, 1, is_terminal, reward)) {
        if (is_terminal != NULL) {
            *is_terminal = 1;
        }
        if (reward != NULL) {
            *reward = -1.0;
        }
        return 0;
    }

    discard_move_undo_captures(next, &undo);
    return 1;
}

static int move_gives_check(CPUBoard *state, int move, char current_player)
{
    CPUMoveUndo undo;
    int gives_check;
    Board *board;

    if (state == NULL) {
        return 0;
    }
    board = &state->board;

    if (!make_move_inplace(state, move, current_player, &undo, 0, NULL, NULL)) {
        return 0;
    }

    gives_check = IsInCheck(board,
                            player_color(opponent_player(current_player)));
    undo_move_inplace(state, &undo);

    return gives_check;
}

// Move ordering for quiescence search. Outside of check evasions, quiescence
// only extends capture sequences to keep the leaf search cheap.
static int is_noisy_move(CPUBoard *state, int move, char current_player)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;
    Piece *moving_piece;
    Piece *target_piece;
    Position target_position;
    Board *board;

    if (state == NULL) {
        return 0;
    }
    board = &state->board;

    decode_move(move, &from_col, &from_row, &to_col, &to_row);
    if (!is_inside_board_coords(from_col, from_row) ||
        !is_inside_board_coords(to_col, to_row)) {
        return 0;
    }

    moving_piece = board->square[from_col][from_row];
    if (moving_piece == NULL) {
        return 0;
    }

    target_piece = board->square[to_col][to_row];
    if (target_piece != NULL && target_piece->color != moving_piece->color) {
        return 1;
    }

    target_position = position_from_coords(to_col, to_row);
    if (moving_piece->type == PAWN &&
        board->en_passant_available &&
        target_position == board->en_passant_target) {
        return 1;
    }

    return 0;
}


static int count_attackers_to_square(Board *board, Position square, Color color)
{
    int count = 0;

    if (board == NULL) {
        return 0;
    }

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = board->square[col][row];

            if (piece == NULL || piece->color != color) {
                continue;
            }

            if (piece_controls_square(board, piece, square)) {
                count++;
            }
        }
    }

    return count;
}

static int should_prune_losing_capture(Board *board, int move, char current_player)
{
    int from_col, from_row, to_col, to_row;
    Piece *moving_piece;
    int moving_value;
    int captured_value;
    int enemy_attackers;
    int friendly_defenders;
    int pawn_value;
    Position target;

    if (board == NULL) {
        return 0;
    }

    decode_move(move, &from_col, &from_row, &to_col, &to_row);
    if (!is_inside_board_coords(from_col, from_row) ||
        !is_inside_board_coords(to_col, to_row)) {
        return 0;
    }

    moving_piece = board->square[from_col][from_row];
    if (moving_piece == NULL) {
        return 0;
    }

    /* Keep qsearch safe: do not prune king moves or weird anteater captures yet. */
    if (moving_piece->type == KING || moving_piece->type == ANTEATER) {
        return 0;
    }

    captured_value = captured_piece_value_for_move(board, move);
    if (captured_value <= 0) {
        return 0;
    }

    moving_value = piece_value_for_phase(moving_piece->type, 0);

    /* If we capture something at least as valuable as ourselves, keep it. */
    if (moving_value <= captured_value) {
        return 0;
    }

    target = position_from_coords(to_col, to_row);

    enemy_attackers = count_attackers_to_square(
        board, target, player_color(opponent_player(current_player)));

    if (enemy_attackers == 0) {
        return 0;
    }

    friendly_defenders = count_attackers_to_square(
        board, target, moving_piece->color);

    /*
        The moving piece itself may be counted as attacking the target square
        before the move. That does NOT mean it is safely defended after capturing.
    */
    if (piece_controls_square(board, moving_piece, target) && friendly_defenders > 0) {
        friendly_defenders--;
    }

    pawn_value = piece_value_for_phase(PAWN, 0);

    /*
        Cheap prune rules:
        1. Hanging expensive capture: lose more than a pawn and square is undefended
        2. Badly outnumbered expensive capture
    */
    if (friendly_defenders == 0 &&
        moving_value >= captured_value + pawn_value) {
        return 1;
    }

    if (enemy_attackers > friendly_defenders &&
        moving_value >= captured_value + 3 * pawn_value) {
        return 1;
    }

    return 0;
}


// Returns a score for move ordering in quiescence search. Higher is better.
// Noisy moves are capture moves only.
static int qsearch_move_order_score(CPUBoard *state, int move, char current_player,
                                    int *is_noisy_out)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;
    Piece *moving_piece;
    int captured_value;
    int score = 0;
    int noisy;
    Board *board;

    
    if (state == NULL) {
        if (is_noisy_out != NULL) {
            *is_noisy_out = 0;
        }
        return 0;
    }
    board = &state->board;

    noisy = is_noisy_move(state, move, current_player);
    if (is_noisy_out != NULL) {
        *is_noisy_out = noisy;
    }
    if (!noisy) {
        return 0;
    }

    decode_move(move, &from_col, &from_row, &to_col, &to_row);
    if (!is_inside_board_coords(from_col, from_row) ||
        !is_inside_board_coords(to_col, to_row)) {
        return 0;
    }

    moving_piece = board->square[from_col][from_row];
    captured_value = captured_piece_value_for_move(board, move);
    if (captured_value > 0) {
        score += 200000 + captured_value * 32;
        if (moving_piece != NULL) {
            score -= piece_value_for_phase(moving_piece->type, 0) / 4;
        }
    }

    if (score <= 0) {
        score = 1000;
    }

    return score;
}

// Orders moves for quiescence search, putting capture moves first. Returns the
// number of capture moves that are kept in the list.
static int order_qsearch_moves(CPUBoard *state, int *moves, int move_count,
                               char current_player)
{
    int noisy_moves[CPU_MAX_MOVES];
    int noisy_scores[CPU_MAX_MOVES];
    int noisy_count = 0;

    if (state == NULL || moves == NULL || move_count <= 0) {
        return 0;
    }
    if (move_count > CPU_MAX_MOVES) {
        move_count = CPU_MAX_MOVES;
    }

    for (int i = 0; i < move_count; i++) {
        int noisy = 0;
        int score = qsearch_move_order_score(state, moves[i], current_player,
                                             &noisy);

        if (!noisy) {
            continue;
        }

        noisy_moves[noisy_count] = moves[i];
        noisy_scores[noisy_count] = score;
        for (int j = noisy_count; j > 0; j--) {
            if (noisy_scores[j] <= noisy_scores[j - 1]) {
                break;
            }

            {
                int swap_move = noisy_moves[j];
                int swap_score = noisy_scores[j];
                noisy_moves[j] = noisy_moves[j - 1];
                noisy_scores[j] = noisy_scores[j - 1];
                noisy_moves[j - 1] = swap_move;
                noisy_scores[j - 1] = swap_score;
            }
        }
        noisy_count++;
    }

    if (noisy_count > 0) {
        memcpy(moves, noisy_moves, sizeof(int) * (size_t)noisy_count);
    }

    return noisy_count;
}

// Quiescence search that extends the search at leaf nodes to avoid horizon
// effect. When the side to move is not in check, it only considers captures.
// Returns a score from the perspective of search_player, where higher is better
// for search_player.

static double qsearch_delta_gain(Board *board, int move)
{
    int captured = captured_piece_value_for_move(board, move);
    return (double)captured / 6500.0;
}
static const double DELTA_MARGIN = 0.03;

static double quiescence_search(CPUBoard *state, char current_player, char search_player,
                                double alpha, double beta, int depth)
{
    int move_count = 0;
    int moves[CPU_MAX_MOVES];
    int maximizing;
    int in_check;
    double stand_pat;
    double best;
    Board *board;

    current_player = normalize_player(current_player);
    search_player = normalize_player(search_player);

    if (state == NULL) {
        return 0.0;
    }
    board = &state->board;

    maximizing = current_player == search_player;
    in_check = IsInCheck(board, player_color(current_player));
    stand_pat = evaluate_position_for_side(board, search_player, current_player); //evaluate from the perspective of search_player, but with current_player to move, which is what we want for quiescence search since it is called from alpha-beta search with the same player to move as the leaf evaluation.
    if (in_check && depth < 0) {
        return stand_pat;
    }

    if (!in_check) {
        if (depth <= 0) {
            return stand_pat;
        }
        if (maximizing) {
            if (stand_pat >= beta) {
                return beta;
            }
            if (stand_pat > alpha) {
                alpha = stand_pat;
            }
        } else {
            if (stand_pat <= alpha) {
                return alpha;
            }
            if (stand_pat < beta) {
                beta = stand_pat;
            }
        }
        best = stand_pat;
    } else {
        best = maximizing ? -1.0e30 : 1.0e30;
    }

    if (in_check) {
        move_count = generate_valid_moves(board, current_player, moves, CPU_MAX_MOVES);
    } else {
        move_count = generate_capture_moves(board, current_player, moves, CPU_MAX_MOVES);
    }
    if (move_count <= 0) {
        if (in_check) {
            return current_player == search_player ? -1.0 : 1.0;
        }
        return stand_pat;
    }

    if (!in_check) {
        move_count = order_qsearch_moves(state, moves, move_count, current_player);
        if (move_count <= 0) {
            return stand_pat;
        }
    }

    for (int i = 0; i < move_count; i++) {
        int is_terminal = 0;
        double reward = 0.0;
        CPUMoveUndo undo;
        double value;

            //delta pruning kinda
           if (!in_check) {
            double gain = qsearch_delta_gain(board, moves[i]);
            // If the move is not a capture that gains enough to raise the score above alpha or below beta, skip it. This is a form of delta pruning that keeps the quiescence search from exploring capture sequences that are unlikely to affect the evaluation enough to change the decision. 

            if (maximizing) {
                if (stand_pat + gain + DELTA_MARGIN <= alpha) {
                    continue;
                }
            } 
            else {
                if (stand_pat - gain - DELTA_MARGIN >= beta) {
                    continue;
                }
            }
        }

        if (!make_move_inplace(state, moves[i], current_player, &undo, 0,
                               &is_terminal, &reward)) {
            continue;
        }
        if (is_terminal) {
            value = current_player == search_player ? reward : -reward;
        } else {
            value = quiescence_search(state, opponent_player(current_player),
                                      search_player, alpha, beta, depth - 1);
        }
        undo_move_inplace(state, &undo);

        if (maximizing) {
            if (value > best) {
                best = value;
            }
            if (best > alpha) {
                alpha = best;
            }
        } else {
            if (value < best) {
                best = value;
            }
            if (best < beta) {
                beta = best;
            }
        }

        if (beta <= alpha) {
            break;
        }
    }

    return clamp_double(best, -1.0, 1.0);
}

// Alpha-beta search that returns a score from the perspective of search_player, where higher is better for search_player. It assumes that both sides play optimally according to the evaluation function and search depth. It does not do any move ordering or pruning beyond what is done in quiescence search, so it is not very efficient on its own. It is used as the leaf evaluation in alpha_beta_playout, which does move ordering and pruning in the quiescence search to keep the leaf search cheap while still avoiding horizon effect.
static double alpha_beta_search(CPUBoard *state, char current_player, char search_player,
                                int depth, double alpha, double beta)
{
    int move_count = 0;
    int moves[CPU_MAX_MOVES];
    int maximizing;
    double best;
    Board *board;

    current_player = normalize_player(current_player);
    search_player = normalize_player(search_player);

    if (state == NULL) {
        return 0.0;
    }
    board = &state->board;
    if (IsCheckmate(board, player_color(current_player))) {
        return current_player == search_player ? -1.0 : 1.0;
    }
    if (depth <= 0) {
        return quiescence_search(state, current_player, search_player, alpha, beta,
                                 CPU_rollout_params.quiescence_depth);
    }

    move_count = generate_valid_moves(board, current_player, moves, CPU_MAX_MOVES);
    if (move_count <= 0) {
        return 0.0;
    }

    maximizing = current_player == search_player;
    best = maximizing ? -1.0e30 : 1.0e30;

    for (int i = 0; i < move_count; i++) {
        int is_terminal = 0;
        double reward = 0.0;
        CPUMoveUndo undo;
        double value;
        //add is in check after
        int in_check = 0;
        in_check = IsInCheck(board, player_color(current_player));

         if (!in_check &&should_prune_losing_capture(board, moves[i], current_player)) {
            continue;
        }

        if (!make_move_inplace(state, moves[i], current_player, &undo, 0,
                               &is_terminal, &reward)) {
            continue;
        }
        if (is_terminal) {
            value = current_player == search_player ? reward : -reward;
        } else {
            value = alpha_beta_search(state, opponent_player(current_player),
                                      search_player, depth - 1, alpha, beta);
        }
        undo_move_inplace(state, &undo);

        if (maximizing) {
            if (value > best) {
                best = value;
            }
            if (best > alpha) {
                alpha = best;
            }
        } else {
            if (value < best) {
                best = value;
            }
            if (best < beta) {
                beta = best;
            }
        }

        if (beta <= alpha) {
            break;
        }
    }

    return clamp_double(best, -1.0, 1.0);
}

static double alpha_beta_playout(const CPUBoard *state, char current_player,
                                 char search_player)
{
    CPUBoard copy;

    copy_search_board(&copy, state);
    double result = alpha_beta_search(&copy, current_player, search_player,
                                      CPU_settings.alpha_beta_depth,
                                      -1.0e30, 1.0e30);
    return result;
}

static int tactical_piece_value(Type type)
{
    if (type == KING) {
        return 20000;
    }

    return piece_value_for_phase(type, 0);
}

static int is_clear_line_to_square(Board *state, int from_col, int from_row,
                                   int to_col, int to_row,
                                   int step_col, int step_row)
{
    int col = from_col + step_col;
    int row = from_row + step_row;

    while (col != to_col || row != to_row) {
        if (!is_inside_board_coords(col, row) ||
            state->square[col][row] != NULL) {
            return 0;
        }
        col += step_col;
        row += step_row;
    }

    return 1;
}

static int piece_controls_square(Board *state, Piece *piece, Position target_position)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;
    int d_col;
    int d_row;
    int abs_col;
    int abs_row;
    Piece *target_piece;

    if (state == NULL || piece == NULL || piece->position == target_position) {
        return 0;
    }

    from_col = FindColumnPosition(piece->position);
    from_row = FindRowPosition(piece->position);
    to_col = FindColumnPosition(target_position);
    to_row = FindRowPosition(target_position);
    if (!is_inside_board_coords(from_col, from_row) ||
        !is_inside_board_coords(to_col, to_row)) {
        return 0;
    }

    d_col = to_col - from_col;
    d_row = to_row - from_row;
    abs_col = abs(d_col);
    abs_row = abs(d_row);
    target_piece = state->square[to_col][to_row];

    switch (piece->type) {
        case PAWN: {
            int direction = piece->color == WHITE ? 1 : -1;
            return abs_col == 1 && d_row == direction;
        }

        case KNIGHT:
            return (abs_col == 2 && abs_row == 1) ||
                   (abs_col == 1 && abs_row == 2);

        case BISHOP:
            if (abs_col != abs_row) {
                return 0;
            }
            return is_clear_line_to_square(state, from_col, from_row, to_col, to_row,
                                           d_col > 0 ? 1 : -1,
                                           d_row > 0 ? 1 : -1);

        case ROOK:
            if (d_col != 0 && d_row != 0) {
                return 0;
            }
            return is_clear_line_to_square(state, from_col, from_row, to_col, to_row,
                                           d_col == 0 ? 0 : (d_col > 0 ? 1 : -1),
                                           d_row == 0 ? 0 : (d_row > 0 ? 1 : -1));

        case QUEEN:
            if (abs_col == abs_row) {
                return is_clear_line_to_square(state, from_col, from_row, to_col, to_row,
                                               d_col > 0 ? 1 : -1,
                                               d_row > 0 ? 1 : -1);
            }
            if (d_col == 0 || d_row == 0) {
                return is_clear_line_to_square(state, from_col, from_row, to_col, to_row,
                                               d_col == 0 ? 0 : (d_col > 0 ? 1 : -1),
                                               d_row == 0 ? 0 : (d_row > 0 ? 1 : -1));
            }
            return 0;

        case KING:
            return abs_col <= 1 && abs_row <= 1;

        case ANTEATER:
            return target_piece != NULL && target_piece->type == PAWN &&
                   target_piece->color != piece->color &&
                   FindAnteaterCapturePath(state, piece,
                                           target_position, NULL, 0) > 0;

        default:
            return 0;
    }
}

static int square_control_count(Board *state, Position target_position, Color color)
{
    int count = 0;

    if (state == NULL) {
        return 0;
    }

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = state->square[col][row];
            if (piece != NULL && piece->color == color &&
                piece_controls_square(state, piece, target_position)) {
                count++;
            }
        }
    }

    return count;
}

static int least_control_value(Board *state, Position target_position, Color color)
{
    int best_value = 0;

    if (state == NULL) {
        return 0;
    }

    for (int col = 0; col < COLUMN; col++) {
        for (int row = 0; row < ROW; row++) {
            Piece *piece = state->square[col][row];
            if (piece != NULL && piece->color == color &&
                piece_controls_square(state, piece, target_position)) {
                int value = tactical_piece_value(piece->type);
                if (best_value == 0 || value < best_value) {
                    best_value = value;
                }
            }
        }
    }

    return best_value;
}


// Returns a penalty score for a piece on the given square based on how unsafe it is. Higher is worse. Only considers pieces that are under attack, not kings.
static int piece_safety_penalty(Board *state, int col, int row, Color color)
{
    Color opponent = color == WHITE ? BLACK : WHITE;
    Piece *piece;
    Position position;
    int attackers;
    int defenders;
    int value;
    int least_attacker;

    if (state == NULL || !is_inside_board_coords(col, row)) {
        return 0;
    }

    piece = state->square[col][row];
    if (piece == NULL || piece->color != color || piece->type == KING) {
        return 0;
    }

    position = position_from_coords(col, row);
    attackers = square_control_count(state, position, opponent);
    if (attackers <= 0) {
        return 0;
    }

    defenders = square_control_count(state, position, color);
    value = tactical_piece_value(piece->type);
    least_attacker = least_control_value(state, position, opponent);

    if (defenders <= 0) {
        return value;
    }
    if (least_attacker > 0 && least_attacker < value) {
        return value / 2;
    }

    return value / 10;
}

static int captured_piece_value_for_move(Board *state, int move)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;
    Piece *moving_piece;
    Piece *captured_piece;
    Position target_position;
    int anteater_capture_count;

    if (state == NULL) {
        return 0;
    }

    decode_move(move, &from_col, &from_row, &to_col, &to_row);
    if (!is_inside_board_coords(from_col, from_row) ||
        !is_inside_board_coords(to_col, to_row)) {
        return 0;
    }

    moving_piece = state->square[from_col][from_row];
    captured_piece = state->square[to_col][to_row];
    target_position = position_from_coords(to_col, to_row);
    if (moving_piece != NULL && moving_piece->type == ANTEATER) {
        anteater_capture_count = FindAnteaterCapturePath(state,
                                                         moving_piece,
                                                         target_position,
                                                         NULL, 0);
        if (anteater_capture_count > 0) {
            return anteater_capture_count * tactical_piece_value(PAWN);
        }
    }

    if (captured_piece != NULL) {
        return tactical_piece_value(captured_piece->type);
    }

    if (moving_piece != NULL &&
        moving_piece->type == PAWN &&
        state->en_passant_available &&
        target_position == state->en_passant_target) {
        int ep_col = FindColumnPosition(state->en_passant_pawn);
        int ep_row = FindRowPosition(state->en_passant_pawn);
        if (is_inside_board_coords(ep_col, ep_row) &&
            state->square[ep_col][ep_row] != NULL) {
            return tactical_piece_value(state->square[ep_col][ep_row]->type);
        }
    }

    return 0;
}


// Returns an adjustment to the static evaluation after making the given move, based on changes in piece safety and potential captures. Higher is better for the moving player.
static int static_exchange_adjustment_after_move(Board *after, int move,
                                                 char current_player,
                                                 int captured_value,
                                                 int before_penalty)
{
    int from_col_unused;
    int from_row_unused;
    int to_col;
    int to_row;
    Color mover_color = player_color(current_player);
    Color opponent_color = mover_color == WHITE ? BLACK : WHITE;
    Position target_position;
    Piece *moved_piece;
    int moved_value;
    int attackers;
    int defenders;
    int least_attacker;
    int after_penalty;
    int adjustment;

    if (after == NULL) {
        return 0;
    }

    decode_move(move, &from_col_unused, &from_row_unused, &to_col, &to_row);
    if (!is_inside_board_coords(to_col, to_row)) {
        return 0;
    }

    target_position = position_from_coords(to_col, to_row);
    moved_piece = after->square[to_col][to_row];
    if (moved_piece == NULL) {
        return 0;
    }

    moved_value = tactical_piece_value(moved_piece->type);
    attackers = square_control_count(after, target_position, opponent_color);
    defenders = square_control_count(after, target_position, mover_color);
    least_attacker = least_control_value(after, target_position, opponent_color);
    after_penalty = piece_safety_penalty(after, to_col, to_row, mover_color);
    adjustment = before_penalty - after_penalty;

    if (attackers > 0) {
        int danger = moved_value - captured_value;
        if (danger < moved_value / 4) {
            danger = moved_value / 4;
        }

        if (defenders <= 0) {
            adjustment -= danger;
        } else if (least_attacker > 0 && least_attacker < moved_value) {
            adjustment -= danger / 2;
        } else {
            adjustment -= moved_value / 10;
        }
    } else if (captured_value > 0) {
        adjustment += captured_value / 10;
    }

    if (IsInCheck(after, opponent_color)) {
        adjustment += 35;
    }

    return adjustment;
}

static int static_exchange_adjustment(Board *before, Board *after, int move,
                                      char current_player)
{
    int from_col;
    int from_row;
    int to_col_unused;
    int to_row_unused;
    Color mover_color = player_color(current_player);
    int captured_value;
    int before_penalty;

    if (before == NULL || after == NULL) {
        return 0;
    }

    decode_move(move, &from_col, &from_row, &to_col_unused, &to_row_unused);

    captured_value = captured_piece_value_for_move(before, move);
    before_penalty = piece_safety_penalty(before, from_col, from_row, mover_color);
    return static_exchange_adjustment_after_move(after, move, current_player,
                                                 captured_value, before_penalty);
}

static int root_tactical_adjustment(Board *before, Board *after, int move,
                                    char current_player)
{
    int adjustment = static_exchange_adjustment(before, after, move, current_player);
    int captured_value = captured_piece_value_for_move(before, move);

    if (captured_value > 0) {
        adjustment += captured_value;
    }

    return adjustment;
}

// Determines whether a move in a rollout should trigger a reply search based on tactical considerations. Returns 1 if a reply search is recommended, 0 otherwise.
static int rollout_move_needs_reply_search(CPUBoard *state, int move,
                                           char current_player,
                                           int tactical_adjustment)
{
    int from_col;
    int from_row;
    int to_col;
    int to_row;
    Piece *piece;
    Board *board;

    if (state == NULL) {
        return 0;
    }
    board = &state->board;

    if (tactical_adjustment < CPU_rollout_params.tactical_reply_threshold ||
        captured_piece_value_for_move(board, move) > 0 ||
        move_gives_check(state, move, current_player)) {
        return 1;
    }

    decode_move(move, &from_col, &from_row, &to_col, &to_row);
    if (!is_inside_board_coords(from_col, from_row) ||
        !is_inside_board_coords(to_col, to_row)) {
        return 0;
    }

    piece = board->square[from_col][from_row];
    return piece != NULL &&
           piece->type == PAWN &&
           ((piece->color == WHITE && to_row == ROW - 1) ||
            (piece->color == BLACK && to_row == 0));
}


// Selects the next move index to consider during rollout move selection, using a combination of capture value and random sampling. Marks the selected move in the sampled array if provided. Returns the index of the selected move, or -1 if no valid moves are available.
static int select_next_rollout_candidate(Board *state, int *moves, int move_count,
                                         int *sampled, int sample_number)
{
    int best_capture_index = -1;
    int best_capture_value = 0;

    if (moves == NULL || move_count <= 0) {
        return -1;
    }

    if (sampled != NULL) {
        for (int i = 0; i < move_count; i++) {
            int capture_value;

            if (sampled[i]) {
                continue;
            }

            capture_value = captured_piece_value_for_move(state, moves[i]);
            if (capture_value > best_capture_value) {
                best_capture_value = capture_value;
                best_capture_index = i;
            }
        }

        if (best_capture_index >= 0) {
            sampled[best_capture_index] = 1;
            return best_capture_index;
        }
    }

    if (sampled != NULL) {
        int unvisited_count = 0;
        int selected = -1;

        for (int i = 0; i < move_count; i++) {
            if (!sampled[i]) {
                unvisited_count++;
                if ((next_random_u32() % (unsigned int)unvisited_count) == 0) {
                    selected = i;
                }
            }
        }

        if (selected >= 0) {
            sampled[selected] = 1;
            return selected;
        }
    }

    (void)sample_number;
    return (int)(next_random_u32() % (unsigned int)move_count);
}

static int choose_rollout_move(CPUBoard *state, char current_player, char search_player,
                               int *moves, int move_count)
{
    int sample_count;
    int best_move;
    double best_score;
    int maximizing;
    int *sampled;
    Board *board;

    if (state == NULL || moves == NULL || move_count <= 0) {
        return -1;
    }
    board = &state->board;

    best_move = moves[(int)(next_random_u32() % (unsigned int)move_count)];
    if (CPU_rollout_params.sample_size <= 0) {
        sample_count = move_count;
    } else {
        sample_count = move_count < CPU_rollout_params.sample_size
            ? move_count
            : CPU_rollout_params.sample_size;
    }
    maximizing = normalize_player(current_player) == normalize_player(search_player);
    best_score = maximizing ? -1.0e30 : 1.0e30;
    sampled = (int *)calloc((size_t)move_count, sizeof(int));

    for (int i = 0; i < sample_count; i++) {
        int move_index;
        int move;
        int from_col;
        int from_row;
        int to_col;
        int to_row;
        int is_terminal = 0;
        double reward = 0.0;
        CPUMoveUndo undo;
        double score;
        int captured_value;
        int before_penalty;

        move_index = select_next_rollout_candidate(board, moves, move_count,
                                                   sampled, i);
        if (move_index < 0) {
            continue;
        }

        move = moves[move_index];
        decode_move(move, &from_col, &from_row, &to_col, &to_row);
        captured_value = captured_piece_value_for_move(board, move);
        before_penalty = piece_safety_penalty(board, from_col, from_row,
                                              player_color(current_player));
        if (!make_move_inplace(state, move, current_player, &undo, 0,
                               &is_terminal, &reward)) {
            continue;
        }
        if (is_terminal) {
            score = maximizing ? reward : -reward;
        } else {
            int tactical_adjustment = static_exchange_adjustment_after_move(
                board, move, current_player, captured_value, before_penalty);
            if (rollout_move_needs_reply_search(state, move, current_player,
                                                tactical_adjustment)) {
                score = alpha_beta_search(state, opponent_player(current_player),
                                          search_player,
                                          CPU_rollout_params.reply_search_depth,
                                          -1.0e30, 1.0e30);
            } else {
                score = evaluate_position_for_side(board, search_player,
                                                   opponent_player(current_player));
            }
            score += (maximizing ? 1.0 : -1.0) *
                     ((double)tactical_adjustment /
                      CPU_rollout_params.tactical_score_scale);
            if (CPU_rollout_params.randomness > 0.0) {
                score += (random_unit() - 0.5) *
                         CPU_rollout_params.randomness * 0.08;
            }
            score = clamp_double(score, -1.0, 1.0);
        }

        undo_move_inplace(state, &undo);

        if ((maximizing && score > best_score) ||
            (!maximizing && score < best_score)) {
            best_score = score;
            best_move = move;
        }
    }

    free(sampled);
    return best_move;
}

// Creates a new tree node for the given game state and player, initializing its fields and allocating memory for its children based on the specified child capacity. Returns a pointer to the newly created node, or NULL if memory allocation fails.
static NewTreeNode *create_node_for_player(const CPUBoard *state, NewTreeNode *parent,
                                           int child_capacity, char player)
{
    NewTreeNode *node = (NewTreeNode *)calloc(1, sizeof(NewTreeNode));
    int capacity;

    if (node == NULL) {
        return NULL;
    }

    copy_search_board(&node->state, state);
    node->parent = parent;
    node->player = normalize_player(player);
    node->untried_moves = NULL;
    node->untried_count = 0;
    node->untried_total = 0;
    node->moves_generated = 0;
    node->heuristic_value = static_evaluate_position_for_side(&node->state.board,
                                                              CPU_starting_player,
                                                              node->player);
    node->implicit_minimax_value = node->heuristic_value;
    node->check_pressure = check_pressure_for_player(&node->state.board,
                                                     CPU_starting_player);
    node->heuristic_ready = 1;
    node->move_that_led_here_encoded = -1;
    node->move_that_led_here[0] = '\0';

    capacity = child_capacity > 0 ? child_capacity : CPU_INITIAL_CHILD_CAPACITY;
    if (capacity < CPU_INITIAL_CHILD_CAPACITY) {
        capacity = CPU_INITIAL_CHILD_CAPACITY;
    }

    node->child_capacity = capacity;
    node->children = (NewTreeNode **)calloc((size_t)capacity, sizeof(NewTreeNode *));
    if (node->children == NULL) {
        free(node->untried_moves);
        free(node);
        return NULL;
    }

    return node;
}

static int ensure_node_moves(NewTreeNode *node)
{
    if (node == NULL) {
        return 0;
    }

    if (!node->moves_generated) {
        node->untried_moves = get_valid_moves(&node->state.board, node->player,
                                              &node->untried_total);
        node->untried_count = node->untried_total;
        node->moves_generated = 1;
    }

    return node->untried_count;
}

static int ensure_child_capacity(NewTreeNode *node)
{
    NewTreeNode **children;
    int new_capacity;

    if (node == NULL) {
        return 0;
    }
    if (node->child_count < node->child_capacity) {
        return 1;
    }

    new_capacity = node->child_capacity > 0
        ? node->child_capacity * 2
        : CPU_INITIAL_CHILD_CAPACITY;
    if (new_capacity <= node->child_count) {
        new_capacity = node->child_count + 1;
    }

    children = (NewTreeNode **)realloc(node->children,
                                       (size_t)new_capacity * sizeof(NewTreeNode *));
    if (children == NULL) {
        return 0;
    }

    for (int i = node->child_capacity; i < new_capacity; i++) {
        children[i] = NULL;
    }
    node->children = children;
    node->child_capacity = new_capacity;
    return 1;
}

static void free_node(NewTreeNode *node)
{
    if (node == NULL) {
        return;
    }

    for (int i = 0; i < node->child_count; i++) {
        free_node(node->children[i]);
    }

    free(node->children);
    free(node->untried_moves);
    free(node);
}

static NewSelectionPolicy default_policy(void)
{
    NewSelectionPolicy policy;
    policy.exploitation_weight = CPU_policy_params.exploitation_weight;
    policy.exploration_weight = CPU_policy_params.exploration_weight;
    policy.prior_weight = CPU_policy_params.prior_weight;
    policy.check_weight = CPU_policy_params.check_weight;
    policy.variance_weight = CPU_policy_params.variance_weight;
    policy.visit_bias = CPU_policy_params.visit_bias;
    policy.implicit_minimax_weight = CPU_policy_params.implicit_minimax_weight;
    return policy;
}

static double child_policy_score(NewTreeNode *parent, NewTreeNode *child, const NewSelectionPolicy *policy)
{
    double exploitation;
    double implicit_minimax;
    double blended_value;
    double exploration;
    double prior;
    double variance = 0.0;
    double check_bonus = 0.0;
    int parent_visits;

    if (parent == NULL || child == NULL || policy == NULL) {
        return -1.0e30;
    }
    if (child->visits == 0) {
        return 1.0e30 + random_unit();
    }

    exploitation = child->total_value / (double)child->visits;
    implicit_minimax = child->implicit_minimax_value;
    if (normalize_player(parent->player) != CPU_starting_player) {
        exploitation = -exploitation;
        implicit_minimax = -implicit_minimax;
    }
    blended_value = (1.0 - policy->implicit_minimax_weight) * exploitation +
                    policy->implicit_minimax_weight * implicit_minimax;

    parent_visits = parent->visits > 1 ? parent->visits : 2;
    exploration = sqrt(log((double)parent_visits) / (double)child->visits);
    prior = child->heuristic_value;
    if (normalize_player(parent->player) != CPU_starting_player) {
        prior = -prior;
    }

    if (child->visits > 1) {
        double mean_square = child->total_square_value / (double)child->visits;
        double mean = child->total_value / (double)child->visits;
        variance = mean_square - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        variance = sqrt(variance);
    }

    check_bonus = child->check_pressure;
    if (normalize_player(parent->player) != CPU_starting_player) {
        check_bonus = -check_bonus;
    }

    return policy->exploitation_weight * blended_value +
           policy->exploration_weight * exploration +
           policy->prior_weight * prior +
           policy->check_weight * check_bonus +
           policy->variance_weight * variance +
           policy->visit_bias / (double)(child->visits + 1);
}

static NewTreeNode *select_child(NewTreeNode *node, const NewSelectionPolicy *policy)
{
    NewTreeNode *best = NULL;
    double best_score = -1.0e30;

    if (node == NULL) {
        return NULL;
    }

    for (int i = 0; i < node->child_count; i++) {
        double score = child_policy_score(node, node->children[i], policy);
        if (best == NULL || score > best_score) {
            best = node->children[i];
            best_score = score;
        }
    }

    return best;
}

static NewTreeNode *expand_node(NewTreeNode *node)
{
    int move_index;
    int move;
    int is_terminal = 0;
    double reward = 0.0;
    CPUBoard next_state;
    NewTreeNode *child;

    if (node == NULL) {
        return NULL;
    }
    ensure_node_moves(node); // This will set untried_count and untried_moves if they haven't been set already.
    if (node->untried_count <= 0 || !ensure_child_capacity(node)) {
        return NULL;
    }

    move_index = (int)(next_random_u32() % (unsigned int)node->untried_count);
    move = node->untried_moves[move_index];
    if (!apply_move(&node->state, move, node->player, &next_state, &is_terminal,
                    &reward)) {
        return NULL;
    }

    child = create_node_for_player(&next_state, node, 0, opponent_player(node->player));
    if (child == NULL) {
        return NULL;
    }

    move_to_string(move, child->move_that_led_here, sizeof(child->move_that_led_here));
    child->move_that_led_here_encoded = move;
    if (is_terminal) {
        free(child->untried_moves);
        child->untried_moves = NULL;
        child->untried_count = 0;
        child->untried_total = 0;
        child->moves_generated = 1;
        child->heuristic_value = node->player == CPU_starting_player ? reward : -reward;
        child->implicit_minimax_value = child->heuristic_value;
        child->heuristic_ready = 1;
    }

    node->children[node->child_count++] = child;
    node->untried_moves[move_index] = node->untried_moves[node->untried_count - 1];
    node->untried_count--;

    (void)reward;
    return child;
}

static double simulate_alpha_beta_playout(const CPUBoard *state, char current_player,
                                          char search_player)
{
    CPUBoard sim;
    double result;
    int depth_limit = CPU_settings.playout_depth;

    if (state == NULL) {
        return 0.0;
    }

    current_player = normalize_player(current_player);
    search_player = normalize_player(search_player);
    if (depth_limit < 0) {
        depth_limit = 0;
    }

    copy_search_board(&sim, state);
    for (int depth = 0; depth < depth_limit; depth++) {
        int move_count = 0;
        int moves[CPU_MAX_MOVES];
        int move;
        int is_terminal = 0;
        double reward = 0.0;
        CPUMoveUndo undo;
        Board *sim_board = &sim.board;

        if (IsCheckmate(sim_board, player_color(current_player))) {
            result = current_player == search_player ? -1.0 : 1.0;
            return result;
        }

        move_count = generate_valid_moves(sim_board, current_player, moves,
                                          CPU_MAX_MOVES);
        if (move_count <= 0) {
            return 0.0;
        }

        move = choose_rollout_move(&sim, current_player, search_player, moves, move_count);
        if (move < 0) {
            return 0.0;
        }

        if (!make_move_inplace(&sim, move, current_player, &undo, 0,
                               &is_terminal, &reward)) {
            return 0.0;
        }
        discard_move_undo_captures(&sim, &undo);

        if (is_terminal) {
            result = current_player == search_player ? reward : -reward;
            return clamp_double(result, -1.0, 1.0);
        }

        current_player = opponent_player(current_player);
    }

    result = alpha_beta_playout(&sim, current_player, search_player);
    return clamp_double(result, -1.0, 1.0);
}

static void update_implicit_minimax(NewTreeNode *node)
{
    double best;

    if (node == NULL || node->child_count <= 0) {
        return;
    }

    best = node->children[0]->implicit_minimax_value;
    for (int i = 1; i < node->child_count; i++) {
        double value = node->children[i]->implicit_minimax_value;
        if (normalize_player(node->player) == CPU_starting_player) {
            if (value > best) {
                best = value;
            }
        } else {
            if (value < best) {
                best = value;
            }
        }
    }

    node->implicit_minimax_value = best;
}

static void backpropagate(NewTreeNode *node, double result)
{
    while (node != NULL) {
        node->visits++;
        node->total_value += result;
        node->total_square_value += result * result;
        update_implicit_minimax(node);
        node = node->parent;
    }
}

static double run_one_simulation(NewTreeNode *root, const NewSelectionPolicy *policy)
{
    NewTreeNode *node = root;
    double result;

    if (node == NULL || policy == NULL) {
        return 0.0;
    }

    ensure_node_moves(node);
    while (node->untried_count == 0 && node->child_count > 0) {
        NewTreeNode *selected = select_child(node, policy);
        if (selected == NULL) {
            break;
        }
        node = selected;
        ensure_node_moves(node);
    }

    if (node->untried_count > 0) {
        NewTreeNode *expanded = expand_node(node);
        if (expanded != NULL) {
            node = expanded;
        }
    }

    result = simulate_alpha_beta_playout(&node->state, node->player, CPU_starting_player);
    backpropagate(node, result);
    return result;
}

static double root_child_side_value(const NewRootChild *child,
                                    const NewSelectionPolicy *policy)
{
    double implicit_weight;
    double heuristic_component;

    if (child == NULL) {
        return 0.0;
    }

    implicit_weight = policy != NULL ? policy->implicit_minimax_weight : 0.35;
    heuristic_component = child->heuristic_value +
                          child->tactical_value * CPU_ROOT_TACTICAL_WEIGHT;
    return clamp_double((1.0 - implicit_weight) * heuristic_component +
                        implicit_weight * child->implicit_minimax_value,
                        -1.0, 1.0);
}

static double root_child_selection_score(const NewRootChild *child,
                                         const NewSelectionPolicy *policy,
                                         int total_visits,
                                         int total_in_flight)
{
    double side_estimate;
    double effective_total_value;
    double effective_total_square_value;
    double exploitation;
    double implicit_minimax;
    double blended_value;
    double exploration;
    double variance = 0.0;
    int effective_visits;
    int parent_visits;

    if (child == NULL || policy == NULL || child->terminal || child->subroot == NULL) {
        return -1.0e30;
    }

    side_estimate = root_child_side_value(child, policy);
    effective_visits = child->visits + child->in_flight; // Treat in-flight simulations as if they were completed for the purpose of selection.
    if (effective_visits <= 0) {
        return 1.0e30 + 0.01 * side_estimate + random_unit();
    }

    // Treat reserved-but-unfinished work as pseudo samples backed by side information.
    effective_total_value = child->total_value + side_estimate * (double)child->in_flight;
    effective_total_square_value = child->total_square_value +
                                   side_estimate * side_estimate *
                                   (double)child->in_flight;
    exploitation = effective_total_value / (double)effective_visits;
    implicit_minimax = child->implicit_minimax_value;
    blended_value = (1.0 - policy->implicit_minimax_weight) * exploitation +
                    policy->implicit_minimax_weight * implicit_minimax;

    parent_visits = total_visits + total_in_flight;
    if (parent_visits <= 1) {
        parent_visits = 2;
    }
    exploration = sqrt(log((double)parent_visits) / (double)effective_visits);

    if (effective_visits > 1) {
        double mean_square = effective_total_square_value / (double)effective_visits;
        variance = mean_square - exploitation * exploitation;
        if (variance < 0.0) {
            variance = 0.0;
        }
        variance = sqrt(variance);
    }

    return policy->exploitation_weight * blended_value +
           policy->exploration_weight * exploration +
           policy->prior_weight * (child->heuristic_value +
                                   child->tactical_value * CPU_ROOT_TACTICAL_WEIGHT) +
           policy->check_weight * child->check_pressure +
           policy->variance_weight * variance +
           policy->visit_bias / (double)(effective_visits + 1);
}

static int select_root_child_for_simulation(NewRootScheduler *scheduler)
{
    int total_visits = 0;
    int total_in_flight = 0;
    int selected = -1;
    double best_score = -1.0e30;

    if (scheduler == NULL || scheduler->children == NULL) {
        return -1;
    }

    for (int i = 0; i < scheduler->child_count; i++) {
        NewRootChild *child = &scheduler->children[i];
        total_visits += child->visits;
        total_in_flight += child->in_flight;
    }

    for (int i = 0; i < scheduler->child_count; i++) {
        NewRootChild *child = &scheduler->children[i];
        double score;

        if (child->terminal || child->subroot == NULL) {
            continue;
        }

        score = root_child_selection_score(child, scheduler->policy,
                                           total_visits, total_in_flight);
        if (selected < 0 || score > best_score) {
            selected = i;
            best_score = score;
        }
    }

    return selected;
}

// Reserves a job for simulating a root child. Returns the index of the reserved child, or -1 if no job could be reserved (e.g., due to budget limits or all children being fully explored).
static int reserve_root_child_job(NewRootScheduler *scheduler) 
{
    int child_index;

    if (scheduler == NULL) {
        return -1;
    }

    pthread_mutex_lock(&scheduler->mutex);
    for (;;) {
        if (scheduler->simulations_reserved >= scheduler->simulation_budget) {
            pthread_mutex_unlock(&scheduler->mutex); 
            // No more simulations should be started under the current budget.
            return -1;
        }

        child_index = select_root_child_for_simulation(scheduler);
        if (child_index >= 0) {
            scheduler->children[child_index].in_flight++;
            scheduler->simulations_reserved++;
            scheduler->active_jobs++;
            pthread_mutex_unlock(&scheduler->mutex);
            return child_index;
        }

        if (scheduler->active_jobs <= 0) {
            pthread_mutex_unlock(&scheduler->mutex);
            return -1;
        }

        pthread_cond_wait(&scheduler->condition, &scheduler->mutex);
    }
}

static void finish_root_child_job(NewRootScheduler *scheduler, int child_index,
                                  double result,
                                  int used_shared_tree,
                                  double latest_implicit_value,
                                  int has_latest_implicit_value)
{
    NewRootChild *child;

    if (scheduler == NULL || child_index < 0 || child_index >= scheduler->child_count) {
        return;
    }

    pthread_mutex_lock(&scheduler->mutex);
    child = &scheduler->children[child_index];

    child->visits++;
    child->total_value += result;
    child->total_square_value += result * result;

    if (has_latest_implicit_value) {
        child->implicit_minimax_value = latest_implicit_value;
    } else if (used_shared_tree && child->subroot != NULL) {
        child->implicit_minimax_value = child->subroot->implicit_minimax_value;
    }

    if (child->in_flight > 0) {
        child->in_flight--;
    }

    scheduler->active_jobs--;
    scheduler->simulations_completed++;
    pthread_cond_broadcast(&scheduler->condition);
    pthread_mutex_unlock(&scheduler->mutex);
}

static double run_independent_root_child_simulation(const NewRootChild *child,
                                                    const NewSelectionPolicy *policy,
                                                    double *implicit_value_out)
{
    NewTreeNode *scratch_root;
    double result;

    if (child == NULL || policy == NULL) {
        return 0.0;
    }

    if (implicit_value_out != NULL) {
        *implicit_value_out = child->implicit_minimax_value;
    }

    scratch_root = create_node_for_player(&child->state, NULL, 0,
                                          child->next_player);
    if (scratch_root == NULL) {
        result = simulate_alpha_beta_playout(&child->state,
                                             child->next_player,
                                             CPU_starting_player);
        if (implicit_value_out != NULL) {
            *implicit_value_out = result;
        }
        return result;
    }

    result = run_one_simulation(scratch_root, policy);

    if (implicit_value_out != NULL) {
        *implicit_value_out = scratch_root->implicit_minimax_value;
    }

    free_node(scratch_root);
    return result;
}

static void *search_worker_thread(void *arg)
{
    NewSearchWorkerTask *task = (NewSearchWorkerTask *)arg;

    if (task == NULL || task->scheduler == NULL) {
        return NULL;
    }

    set_thread_seed(task->seed);

    for (;;) {
        int child_index = reserve_root_child_job(task->scheduler);
        NewRootChild *child;
        double result;
        double implicit_value = 0.0;

        if (child_index < 0) {
            break;
        }

        child = &task->scheduler->children[child_index];

        // Attempt to acquire the tree mutex for the child. If successful, run the simulation using the shared tree. If not, run an independent simulation without the shared tree.
        if (child->tree_mutex_ready &&
            pthread_mutex_trylock(&child->tree_mutex) == 0) {

            result = run_one_simulation(child->subroot, task->scheduler->policy);

            finish_root_child_job(task->scheduler,
                                  child_index,
                                  result,
                                  1,
                                  child->subroot != NULL ? child->subroot->implicit_minimax_value : 0.0,
                                  1);

            pthread_mutex_unlock(&child->tree_mutex);
        } else {
            result = run_independent_root_child_simulation(child,
                                                           task->scheduler->policy,
                                                           &implicit_value);

            finish_root_child_job(task->scheduler,
                                  child_index,
                                  result,
                                  0,
                                  implicit_value,
                                  1);
        }
    }

    return NULL;
}

static int initialize_root_children(CPUBoard *root_state, const int *root_moves,
                                    int root_move_count, NewRootChild *children)
{
    Board *root_board;

    if (root_state == NULL || root_moves == NULL || children == NULL ||
        root_move_count <= 0) {
        return 0;
    }
    root_board = &root_state->board;

    for (int i = 0; i < root_move_count; i++) {
        int is_terminal = 0;
        double reward = 0.0;
        CPUBoard child_state;
        Board *child_board;

        memset(&children[i], 0, sizeof(children[i]));
        if (pthread_mutex_init(&children[i].tree_mutex, NULL) != 0) {
            return 0;
        }
        children[i].tree_mutex_ready = 1;
        children[i].move = root_moves[i];
        children[i].next_player = opponent_player(CPU_starting_player);
        if (!apply_move(root_state, root_moves[i], CPU_starting_player,
                        &child_state, &is_terminal, &reward)) {
            return 0;
        }
        child_board = &child_state.board;
        copy_search_board(&children[i].state, &child_state);
        children[i].heuristic_value = static_evaluate_position_for_side(child_board,
                                                                        CPU_starting_player,
                                                                        children[i].next_player);
        children[i].tactical_value = clamp_double(
            (double)root_tactical_adjustment(root_board, child_board,
                                             root_moves[i],
                                             CPU_starting_player) / 6500.0,
            -0.4, 0.4);
        children[i].implicit_minimax_value = children[i].heuristic_value;
        children[i].check_pressure = check_pressure_for_player(child_board,
                                                               CPU_starting_player);

        if (is_terminal) {
            children[i].terminal = 1;
            children[i].visits = 1;
            children[i].total_value = reward;
            children[i].total_square_value = reward * reward;
            children[i].heuristic_value = reward;
            children[i].tactical_value = reward;
            children[i].implicit_minimax_value = reward;
            continue;
        }

        if (children[i].tactical_value > 0.03) {
            double seeded = clamp_double(children[i].heuristic_value +
                                         children[i].tactical_value,
                                         -1.0, 1.0);
            children[i].prior_visits = CPU_CAPTURE_PRIOR_VISITS;
            children[i].prior_total_value = seeded * (double)children[i].prior_visits;
            children[i].prior_total_square_value = seeded * seeded *
                                                   (double)children[i].prior_visits;
            children[i].visits = children[i].prior_visits;
            children[i].total_value = children[i].prior_total_value;
            children[i].total_square_value = children[i].prior_total_square_value;
            children[i].implicit_minimax_value = seeded;
        }

        children[i].subroot = create_node_for_player(&child_state, NULL, 0,
                                                     children[i].next_player);
        if (children[i].subroot == NULL) {
            children[i].terminal = 1;
            children[i].visits = 1;
            children[i].total_value = -1.0;
            children[i].total_square_value = 1.0;
            children[i].implicit_minimax_value = -1.0;
        }
    }

    return 1;
}

static double root_child_final_score(const NewRootChild *child)
{
    double average;

    if (child == NULL || child->visits <= 0) {
        return -1.0e30;
    }

    average = child->total_value / (double)child->visits;
    return average +
           0.35 * child->implicit_minimax_value +
           child->tactical_value * CPU_ROOT_TACTICAL_WEIGHT;
}

static int best_root_child_move(const NewRootChild *children, int child_count)
{
    int best_index = -1;
    double best_score = -1.0e30;

    if (children == NULL || child_count <= 0) {
        return -1;
    }

    for (int i = 0; i < child_count; i++) {
        double score = root_child_final_score(&children[i]);

        if (children[i].visits <= 0) {
            continue;
        }
        if (best_index < 0 || score > best_score ||
            (score == best_score && children[i].visits > children[best_index].visits)) {
            best_index = i;
            best_score = score;
        }
    }

    return best_index >= 0 ? children[best_index].move : -1;
}

static void destroy_root_children(NewRootChild *children, int child_count)
{
    if (children == NULL) {
        return;
    }

    for (int i = 0; i < child_count; i++) {
        free_node(children[i].subroot);
        children[i].subroot = NULL;
        if (children[i].tree_mutex_ready) {
            pthread_mutex_destroy(&children[i].tree_mutex);
            children[i].tree_mutex_ready = 0;
        }
    }
}

static int run_parallel_root_search(CPUBoard *root_state, const NewSelectionPolicy *policy)
{
    pthread_t threads[CPU_THREADS_MAX];
    NewSearchWorkerTask tasks[CPU_THREADS_MAX];
    int started[CPU_THREADS_MAX];
    int *root_moves;
    int root_move_count = 0;
    NewRootChild *children;
    NewRootScheduler scheduler;
    int thread_count = CPU_settings.thread_count;
    int total_simulations = CPU_settings.simulations;
    int best_move;

    if (root_state == NULL || policy == NULL) {
        return -1;
    }

    root_moves = get_valid_moves(&root_state->board, CPU_starting_player, &root_move_count);
    if (root_moves == NULL || root_move_count <= 0) {
        free(root_moves);
        return -1;
    }
    if (total_simulations < root_move_count) {
        total_simulations = root_move_count;
    }

    children = (NewRootChild *)calloc((size_t)root_move_count, sizeof(NewRootChild));
    if (children == NULL) {
        free(root_moves);
        return -1;
    }

    if (!initialize_root_children(root_state, root_moves, root_move_count, children)) {
        destroy_root_children(children, root_move_count);
        free(children);
        free(root_moves);
        return -1;
    }

    if (thread_count < 1) {
        thread_count = 1;
    }
    if (thread_count > CPU_THREADS_MAX) {
        thread_count = CPU_THREADS_MAX;
    }
    if (thread_count > total_simulations) {
        thread_count = total_simulations;
    }

    memset(&scheduler, 0, sizeof(scheduler)); // Clear the scheduler structure before use
    scheduler.children = children;
    scheduler.child_count = root_move_count;
    scheduler.simulation_budget = total_simulations;
    scheduler.policy = policy;
    pthread_mutex_init(&scheduler.mutex, NULL);
    pthread_cond_init(&scheduler.condition, NULL);

    memset(started, 0, sizeof(started)); // Track which threads were successfully started

/* Launch N-1 worker threads */
for (int i = 0; i < thread_count - 1; i++) {

    memset(&tasks[i], 0, sizeof(tasks[i])); // Clear the task structure before use
    tasks[i].scheduler = &scheduler;
    tasks[i].seed = next_random_u32() ^
                    (unsigned int)(0x85ebca6bu + i * 0xc2b2ae35u); 
                    

    // Try to start a thread for this worker task. If thread creation fails, we'll just run it in the main thread.
    if (pthread_create(&threads[i], NULL, search_worker_thread, &tasks[i]) == 0) {
        started[i] = 1;
    } else {
        /* fallback: run this worker inline if thread creation fails */
        search_worker_thread(&tasks[i]);
    }
}

/* Use the main thread as the final worker */
if (thread_count > 0) {
    int i = thread_count - 1;
    memset(&tasks[i], 0, sizeof(tasks[i]));
    tasks[i].scheduler = &scheduler;
    tasks[i].seed = next_random_u32() ^
                    (unsigned int)(0x85ebca6bu + i * 0xc2b2ae35u);

    search_worker_thread(&tasks[i]);
}

/* Join only the pthreads we actually launched */
for (int i = 0; i < thread_count - 1; i++) {
    if (started[i]) {
        pthread_join(threads[i], NULL);
    }
}

    best_move = best_root_child_move(children, root_move_count);
    if (best_move < 0 && root_move_count > 0) {
        best_move = root_moves[0];
    }

    pthread_cond_destroy(&scheduler.condition);
    pthread_mutex_destroy(&scheduler.mutex);
    destroy_root_children(children, root_move_count);
    free(children);
    free(root_moves);
    return best_move;
}

static void configure_settings(const char *difficulty_mode)
{
    if (difficulty_mode != NULL && strcmp(difficulty_mode, "CUSTOM") == 0) {
        return;
    }

    if (difficulty_mode != NULL && strcmp(difficulty_mode, "EASY") == 0) {
        CPU_settings.simulations = CPU_EASY_SIMULATIONS;
        CPU_settings.playout_depth = CPU_EASY_DEPTH;
        CPU_settings.alpha_beta_depth = CPU_EASY_ALPHA_BETA_DEPTH;
        CPU_settings.thread_count = CPU_EASY_THREADS;
    } else if (difficulty_mode != NULL && strcmp(difficulty_mode, "HARD") == 0) {
        CPU_settings.simulations = CPU_HARD_SIMULATIONS;
        CPU_settings.playout_depth = CPU_HARD_DEPTH;
        CPU_settings.alpha_beta_depth = CPU_HARD_ALPHA_BETA_DEPTH;
        CPU_settings.thread_count = CPU_HARD_THREADS;
    } else {
        CPU_settings.simulations = CPU_MEDIUM_SIMULATIONS;
        CPU_settings.playout_depth = CPU_MEDIUM_DEPTH;
        CPU_settings.alpha_beta_depth = CPU_MEDIUM_ALPHA_BETA_DEPTH;
        CPU_settings.thread_count = CPU_MEDIUM_THREADS;
    }
}




//Main function to call for the CPU move, returns the best move found in bestMove as a string (e.g., "e2e4")
//How to use:
//initalize something like char bestMove[10]
//call CPUTreeSearch(currentBoard, currentPlayer, difficultyMode, bestMove)
//after the call, bestMove will contain the best move found by the CPU, or "NONE" if no move is found

void CPUTreeSearch(Board currentState, char player, const char *difficultyMode,
                      char *bestMove, size_t bestMoveSize)
{
    //, GameTimer *timer
    CPUBoard root_state;
    NewSelectionPolicy policy;
    int best_move;
    static unsigned int seed_counter = 0;
    //int time_remaining = GetTimeRemaining()

    if (bestMove != NULL && bestMoveSize > 0) {
        snprintf(bestMove, bestMoveSize, "NONE");
    }

    seed_counter += 0x9e3779b9u; // Increment the seed counter with a large constant to ensure different seeds across calls, even if they happen in the same second.
    set_thread_seed((unsigned int)time(NULL) ^ seed_counter);

    configure_settings(difficultyMode);
    CPU_starting_player = normalize_player(player);

    init_search_board_from_board(&root_state, &currentState);
    if (IsCheckmate(&root_state.board, player_color(CPU_starting_player))) {
        return;
    }

    policy = default_policy();
    best_move = run_parallel_root_search(&root_state, &policy);
    if (bestMove != NULL && bestMoveSize > 0 && best_move >= 0) {
        move_to_string(best_move, bestMove, bestMoveSize);
    }
}
