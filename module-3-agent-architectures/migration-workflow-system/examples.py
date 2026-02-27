"""
Example migrations demonstrating the system.
"""

import json
from pathlib import Path
from src.agent import MigrationAgent
from src.llm_client import LLMClient
from src.state import MigrationState


# Example 1: Express.js to FastAPI migration
EXPRESS_TO_FASTAPI = {
    "source_framework": "express",
    "target_framework": "fastapi",
    "files": {
        "server.js": """const express = require('express');
const app = express();

app.use(express.json());

// Get all users
app.get('/api/users', async (req, res) => {
    try {
        const users = await db.query('SELECT * FROM users');
        res.json(users);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Create user
app.post('/api/users', async (req, res) => {
    const { name, email } = req.body;
    try {
        const result = await db.query(
            'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *',
            [name, email]
        );
        res.status(201).json(result[0]);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});""",
    },
}

# Example 2: Flask to FastAPI migration
FLASK_TO_FASTAPI = {
    "source_framework": "flask",
    "target_framework": "fastapi",
    "files": {
        "app.py": """from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample data
items = []

@app.route('/api/items', methods=['GET'])
def get_items():
    return jsonify(items)

@app.route('/api/items', methods=['POST'])
def create_item():
    data = request.get_json()
    item = {
        'id': len(items) + 1,
        'title': data.get('title'),
        'description': data.get('description')
    }
    items.append(item)
    return jsonify(item), 201

@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    for item in items:
        if item['id'] == item_id:
            return jsonify(item)
    return {'error': 'Not found'}, 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)""",
    },
}

# Example 3: Vue.js to React migration
VUE_TO_REACT = {
    "source_framework": "vue",
    "target_framework": "react",
    "files": {
        "TodoList.vue": """<template>
  <div class="todo-list">
    <h1>{{ title }}</h1>
    <input 
      v-model="newTodo" 
      @keyup.enter="addTodo"
      placeholder="Add a new todo"
    />
    <button @click="addTodo">Add</button>
    <ul>
      <li 
        v-for="todo in todos" 
        :key="todo.id"
        @click="toggleComplete(todo.id)"
        :class="{ completed: todo.completed }"
      >
        {{ todo.text }}
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  data() {
    return {
      title: 'My Todo List',
      newTodo: '',
      todos: [
        { id: 1, text: 'Learn Vue', completed: false },
        { id: 2, text: 'Build something', completed: false }
      ]
    }
  },
  methods: {
    addTodo() {
      if (this.newTodo.trim()) {
        this.todos.push({
          id: Date.now(),
          text: this.newTodo,
          completed: false
        });
        this.newTodo = '';
      }
    },
    toggleComplete(id) {
      const todo = this.todos.find(t => t.id === id);
      if (todo) todo.completed = !todo.completed;
    }
  }
}
</script>

<style>
.completed { text-decoration: line-through; opacity: 0.5; }
</style>""",
    },
}


def run_example(name: str, example: dict) -> None:
    """Run an example migration."""
    print(f"\n{'='*70}")
    print(f"📚 EXAMPLE: {name}")
    print(f"{'='*70}\n")

    llm_client = LLMClient()
    agent = MigrationAgent(llm_client)

    state = MigrationState(
        source_framework=example["source_framework"],
        target_framework=example["target_framework"],
        source_files=example["files"],
    )

    print(f"Migrating from {state.source_framework} to {state.target_framework}...")
    state = agent.run(state)

    print(f"\n✓ Migration complete!")
    print(f"  Success: {len(state.errors) == 0}")
    print(f"  Files generated: {len(state.migrated_files)}")
    print(f"  Errors: {len(state.errors)}")

    if state.migrated_files:
        print(f"\n📄 Generated files:")
        for filename in state.migrated_files:
            print(f"   - {filename}")

    if state.errors:
        print(f"\n⚠️  Errors:")
        for error in state.errors:
            print(f"   - {error}")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("🚀 Migration Workflow System - Example Migrations")
    print("=" * 70)
    print("\nThis script demonstrates the migration system with examples.")
    print("Note: These are demonstration examples. In production, ensure")
    print("that OPENAI_API_KEY is set in your environment.\n")

    examples = [
        ("Express.js to FastAPI", EXPRESS_TO_FASTAPI),
        ("Flask to FastAPI", FLASK_TO_FASTAPI),
        ("Vue.js to React", VUE_TO_REACT),
    ]

    # You can run specific examples by uncommenting:
    # for name, example in examples:
    #     run_example(name, example)

    print("\n📝 To run examples programmatically:")
    print("   from examples import run_example, EXPRESS_TO_FASTAPI")
    print("   run_example('Express to FastAPI', EXPRESS_TO_FASTAPI\n")
