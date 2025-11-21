import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import chess.engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewGameRequest(BaseModel):
    elo: int = 1200

class MoveRequest(BaseModel):
    fen: str
    move: str  # in UCI (e2e4) or SAN (e4)
    ai_elo: int

@app.get("/")
def read_root():
    return {"message": "Chess AI Backend Ready"}

@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    # Start from the initial position, return FEN
    board = chess.Board()

    # Basic difficulty metadata based on ELO buckets
    level = elo_to_level(req.elo)
    return {
        "fen": board.fen(),
        "ai_elo": req.elo,
        "level": level,
        "legal_moves": [m.uci() for m in board.legal_moves]
    }

@app.post("/api/move")
def player_move(req: MoveRequest):
    # Validate board
    try:
        board = chess.Board(req.fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    # Parse player move in either SAN or UCI
    move = None
    try:
        # Try SAN first
        move = board.parse_san(req.move)
    except Exception:
        try:
            move = chess.Move.from_uci(req.move)
            if move not in board.legal_moves:
                raise ValueError("Illegal move")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid or illegal move")

    board.push(move)

    # If game ended after player's move
    if board.is_game_over():
        return {
            "fen": board.fen(),
            "result": game_result(board),
            "ai_move": None,
            "legal_moves": []
        }

    # AI move based on ELO difficulty (simple heuristic engine)
    ai_move = choose_ai_move(board, req.ai_elo)
    board.push(ai_move)

    return {
        "fen": board.fen(),
        "ai_move": ai_move.uci(),
        "result": game_result(board) if board.is_game_over() else None,
        "legal_moves": [m.uci() for m in board.legal_moves]
    }

@app.get("/api/validate-move")
def validate_move(fen: str, move: str):
    try:
        board = chess.Board(fen)
        try:
            mv = board.parse_san(move)
        except Exception:
            mv = chess.Move.from_uci(move)
        if mv not in board.legal_moves:
            raise ValueError("Illegal move")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def elo_to_level(elo: int) -> int:
    # Map ELO to a search depth level 1-4
    if elo < 800:
        return 1
    if elo < 1200:
        return 2
    if elo < 1600:
        return 3
    return 4


def choose_ai_move(board: chess.Board, elo: int) -> chess.Move:
    """
    Lightweight AI:
    - Sample from legal moves with quality-weighted probabilities depending on ELO
    - For higher ELO, perform shallow minimax lookahead using material evaluation
    """
    import random

    level = elo_to_level(elo)
    legal = list(board.legal_moves)

    # For level 1: random legal
    if level == 1:
        return random.choice(legal)

    # Simple evaluation: material count
    def evaluate(b: chess.Board) -> int:
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        score = 0
        for piece_type, val in piece_values.items():
            score += len(b.pieces(piece_type, chess.WHITE)) * val
            score -= len(b.pieces(piece_type, chess.BLACK)) * val
        return score if b.turn == chess.WHITE else -score

    # For level 2: pick best capture if available, else random
    if level == 2:
        captures = [m for m in legal if board.is_capture(m)]
        if captures:
            best = max(captures, key=lambda m: capture_gain(board, m))
            return best
        return random.choice(legal)

    # For level 3-4: shallow minimax with depth = level
    depth = level

    def minimax(b: chess.Board, d: int, alpha: int, beta: int, maximizing: bool) -> int:
        if d == 0 or b.is_game_over():
            return evaluate(b)
        if maximizing:
            max_eval = -10**9
            for m in b.legal_moves:
                b.push(m)
                val = minimax(b, d-1, alpha, beta, False)
                b.pop()
                if val > max_eval:
                    max_eval = val
                if val > alpha:
                    alpha = val
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = 10**9
            for m in b.legal_moves:
                b.push(m)
                val = minimax(b, d-1, alpha, beta, True)
                b.pop()
                if val < min_eval:
                    min_eval = val
                if val < beta:
                    beta = val
                if beta <= alpha:
                    break
            return min_eval

    best_move = None
    best_val = -10**9
    for m in legal:
        board.push(m)
        val = minimax(board, depth-1, -10**9, 10**9, False)
        board.pop()
        if val > best_val:
            best_val = val
            best_move = m
    return best_move or random.choice(legal)


def capture_gain(b: chess.Board, m: chess.Move) -> int:
    victim = b.piece_at(m.to_square)
    if not victim:
        return 0
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }
    return values.get(victim.piece_type, 0)


@app.get("/test")
def test_database():
    """Preserved original diagnostics"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
