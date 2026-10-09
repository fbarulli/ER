comprehensive reference guide to classic software engineering design patterns, architectural structures, and best practices.

---

## 🏗️ 1. Creational Patterns
Creational patterns abstract the instantiation process. They make a system independent of how its objects are created, composed, and represented.

*   **Singleton**
    *   *Intent:* Ensures a class has only one single instance throughout the application lifecycle and provides a global access point to it.
    *   *Use Case:* Managing a shared database connection pool or global application configuration.
*   **Factory Method**
    *   *Intent:* Defines an interface for creating an object, but lets subclasses decide which class to instantiate.
    *   
*   **Abstract Factory**
    *   *Intent:* Provides an interface for creating families of related or dependent objects without specifying their concrete classes.
    *   *Use Case:* Generating complete custom themed toolkits (Dark Mode vs. Light Mode) containing buttons, checkboxes, and text fields together.
*   **Builder**
    *   *Intent:* Separates the construction of a complex object from its representation so that the same construction process can create different representations.
    *   *Use Case:* Building complex objects step-by-step, like an HTTP request builder with optional headers, query params, and body payloads.
*   **Prototype**
    *   *Intent:* Specifies the kinds of objects to create using a prototypical instance, and creates new objects by copying this prototype.
    *   *Use Case:* Instantiating costly objects where copying an existing configured instance is faster than making a fresh database/network lookup.

---

## 🧱 2. Structural Patterns
Structural patterns deal with how classes and objects are composed to form larger structures while keeping these structures flexible and efficient.

*   **Adapter**
    *   *Intent:* Converts the interface of a class into another interface clients expect, letting incompatible classes work together.
    *   *Use Case:* Wrapping a third-party legacy API format (XML) so it satisfies your modern application's internal data interface (JSON).
*   **Facade**
    *   *Intent:* Provides a unified, simplified interface to a set of interfaces in a subsystem, making the subsystem easier to use.
    *   *Use Case:* Exposing a single `orderFlow.checkout()` method that orchestrates inventory checks, payment processing, and shipping notifications behind the scenes.
*   **Decorator**
    *   *Intent:* Attaches additional responsibilities to an object dynamically, providing a flexible alternative to subclassing for extending functionality.
    *   *Use Case:* Wrapping an encryption layer or a logging layer around a standard file-stream reader object.
*   **Proxy**
    *   *Intent:* Provides a surrogate or placeholder for another object to control access to it.
    *   *Use Case:* Implementing lazy loading, access control checks, or caching networks results before reaching the heavy service object.
*   **Composite**
    *   *Intent:* Composes objects into tree structures to represent part-whole hierarchies, letting clients treat individual objects and compositions uniformly.
    *   *Use Case:* Representing file systems containing both files (leafs) and folders (composites containing files or folders).

---

## 🔄 3. Behavioral Patterns
Behavioral patterns are specifically concerned with communication, coordination, and the assignment of responsibilities between objects.

*   **Observer**
    *   *Intent:* Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified automatically.
    *   *Use Case:* Event-driven systems, like notifying multiple UI components when an underlying application state updates.
*   **Strategy**
    *   *Intent:* Defines a family of algorithms, encapsulates each one, and makes them interchangeable at runtime.
    *   *Use Case:* Selecting different payment processing routes  dynamically based on user checkout choices.
*   **State**
    *   *Intent:* Allows an object to alter its behavior when its internal state changes, appearing to change its class.
    *   *Use Case:* Modeling a vending machine or an order tracking system (Ordered ➔ Shipped ➔ Delivered) without nested `if/else` checks.
*   **Iterator**
    *   *Intent:* Provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
    *   *Use Case:* Traversing data collections (lists, trees, graphs) via a standardized loop format.
*   **Command**
    *   *Intent:* Encapsulates a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
    *   *Use Case:* Implementing text editor features like Copy, Paste, and Undo operations.

---

## 🏛️ 4. High-Level Architectural Patterns
While design patterns focus on low-level class layouts, architectural patterns govern the structural blueprint of entire systems.

*   **Model-View-Controller (MVC):** Splits an application into three main components: Model (data), View (UI), and Controller (logic) to separate user interfaces from underlying data.
*   **Repository Pattern:** Abstracts the data persistence layer, providing a collection-like interface for accessing domain data without exposing database-specific operations.
*   **Dependency Injection (DI):** Passes dependent objects into a class rather than allowing the class to instantiate them itself, promoting high decoupleability and testing ease.
