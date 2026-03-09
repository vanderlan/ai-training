// Sample TypeScript file for RAG testing - User Service

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'guest';
  createdAt: Date;
}

export class UserService {
  private users: Map<string, User> = new Map();

  /**
   * Create a new user
   * @param email - User email address
   * @param name - User's full name
   * @param role - User role
   * @returns The created user object
   */
  async createUser(email: string, name: string, role: User['role'] = 'user'): Promise<User> {
    const user: User = {
      id: this.generateId(),
      email,
      name,
      role,
      createdAt: new Date()
    };

    this.users.set(user.id, user);
    return user;
  }

  /**
   * Find a user by ID
   * @param id - User ID
   * @returns User object or undefined if not found
   */
  async findById(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  /**
   * Find users by role
   * @param role - User role to filter by
   * @returns Array of matching users
   */
  async findByRole(role: User['role']): Promise<User[]> {
    const result: User[] = [];
    for (const user of this.users.values()) {
      if (user.role === role) {
        result.push(user);
      }
    }
    return result;
  }

  /**
   * Update user information
   * @param id - User ID
   * @param updates - Partial user object with updates
   * @returns Updated user or undefined if not found
   */
  async updateUser(id: string, updates: Partial<User>): Promise<User | undefined> {
    const user = this.users.get(id);
    if (!user) return undefined;

    const updated = { ...user, ...updates };
    this.users.set(id, updated);
    return updated;
  }

  /**
   * Delete a user
   * @param id - User ID
   * @returns True if deleted, false if not found
   */
  async deleteUser(id: string): Promise<boolean> {
    return this.users.delete(id);
  }

  private generateId(): string {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}
